"""Athlete Lens API + static production UI."""
from __future__ import annotations

import csv
import io
import logging
import threading
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from . import models, schemas
from .config import settings
from .database import Base, SessionLocal, engine
from .ml_engine import SPORTS, engine_singleton
from .seed import seed_if_empty
from .services import (
    PROFILE_FIELDS,
    SESSION_FIELDS,
    athlete_out,
    latest_predictions,
    payload_for,
    score_session,
    session_counts,
    trend,
)

log = logging.getLogger("athlete_lens")
logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "frontend" / "dist"
MAX_CSV_ROWS = 2000

# --------------------------------------------------------------------------- bootstrap
_ready = False
_ready_lock = threading.Lock()


def ensure_ready() -> None:
    """Create tables, load models and seed demo data exactly once per process.

    Called from the ASGI lifespan *and* lazily from request dependencies, because
    serverless runtimes may start a fresh instance (with an empty /tmp) at any time.
    """
    global _ready
    if _ready:
        return
    with _ready_lock:
        if _ready:
            return
        Base.metadata.create_all(bind=engine)
        engine_singleton.load()
        if settings.seed_demo_data:
            with SessionLocal() as db:
                seed_if_empty(db)
        _ready = True
        log.info("Athlete Lens ready (models from %s)", engine_singleton.source)


def get_db():
    ensure_ready()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_ready()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Machine-learning performance, readiness and injury-risk analysis for grassroots athletes.",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- errors
@app.exception_handler(RequestValidationError)
async def validation_handler(_request: Request, exc: RequestValidationError):
    errors = []
    for e in exc.errors():
        loc = ".".join(str(p) for p in e.get("loc", []) if p not in ("body", "query", "path"))
        errors.append({"field": loc or None, "message": e.get("msg"), "type": e.get("type")})
    first = errors[0] if errors else {"field": None, "message": "Invalid request"}
    message = f"{first['field']}: {first['message']}" if first.get("field") else first["message"]
    return JSONResponse(status_code=422, content={"detail": message, "errors": errors})


@app.exception_handler(Exception)
async def unhandled_handler(_request: Request, exc: Exception):  # pragma: no cover - safety net
    log.exception("unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def _get_athlete(db: Session, athlete_id: int) -> models.Athlete:
    athlete = db.get(models.Athlete, athlete_id)
    if athlete is None:
        raise HTTPException(status_code=404, detail=f"Athlete {athlete_id} not found")
    return athlete


# --------------------------------------------------------------------------- health / meta
@app.get("/health", tags=["meta"])
@app.get("/api/health", tags=["meta"])
def health(db: Session = Depends(get_db)):
    m = engine_singleton.meta.get("metrics", {})
    return {
        "ok": True,
        "app": settings.app_name,
        "version": settings.version,
        "env": settings.app_env,
        "platform": settings.platform,
        "athletes": db.query(models.Athlete).count(),
        "model": {
            "ready": engine_singleton.ready,
            "source": "bundled" if "artifacts" in engine_singleton.source else "runtime",
            "sklearn_version": engine_singleton.meta.get("sklearn_version"),
            "injury_holdout_roc_auc": m.get("injury", {}).get("holdout_roc_auc"),
            "performance_holdout_mae": m.get("performance", {}).get("holdout_mae"),
        },
    }


@app.get("/api/sports", tags=["meta"])
def sports():
    return SPORTS


@app.get("/api/model", tags=["meta"])
def model_card():
    engine_singleton.load()
    meta = engine_singleton.meta
    return {
        "name": "Athlete Lens v1",
        "injury_model": meta.get("injury_model"),
        "performance_model": meta.get("performance_model"),
        "training_data": {
            "kind": "synthetic",
            "description": "Physiologically-plausible synthetic cohort of grassroots athletes (ages 13-27, 8 sports). "
                           "Labels come from a hand-specified risk/performance generator, not real injury records.",
            "cohort_size": meta.get("cohort_size"),
            "train_rows": meta.get("train_rows"),
            "test_rows": meta.get("test_rows"),
            "seed": meta.get("seed"),
        },
        "metrics": meta.get("metrics", {}),
        "trained_at": meta.get("trained_at"),
        "sklearn_version": meta.get("sklearn_version"),
        "features": meta.get("features", []),
        "feature_importance": engine_singleton.importance(12),
        "risk_bands": {
            "high": "probability >= 0.55, or <= 1 rest day with RPE >= 8",
            "moderate": "probability >= 0.32",
            "low": "otherwise",
        },
        "readiness_bands": {"green": ">= 65", "amber": "45-64", "red": "< 45"},
        "intended_use": "Coach-facing load and recovery triage for school, college and district athletes "
                        "without lab equipment.",
        "limitations": [
            "Trained on synthetic data: metrics measure how well the model recovers the simulator, not real-world accuracy.",
            "Injury model is only modestly better than chance on held-out data (see ROC AUC); treat bands as prompts to talk, not predictions.",
            "Not a medical device. A high band means 'check with the physio / reduce load', never a diagnosis.",
            "Retrain on 8-12 weeks of real, labelled club sessions before relying on probabilities.",
        ],
    }


# --------------------------------------------------------------------------- dashboard
@app.get("/api/stats", response_model=schemas.DashboardStats, tags=["dashboard"])
def stats(db: Session = Depends(get_db)):
    athletes = db.scalars(select(models.Athlete)).all()
    latest = latest_predictions(db)
    bands = {"low": 0, "moderate": 0, "high": 0, "none": 0}
    readiness = []
    by_sport: dict[str, int] = {}
    watch = []
    for a in athletes:
        by_sport[a.sport] = by_sport.get(a.sport, 0) + 1
        pred, last = latest.get(a.id, (None, None))
        if pred is None:
            bands["none"] += 1
            continue
        bands[pred.injury_risk] = bands.get(pred.injury_risk, 0) + 1
        readiness.append(pred.readiness_score)
        watch.append({
            "athlete_id": a.id, "name": a.name, "sport": a.sport, "injury_risk": pred.injury_risk,
            "injury_probability": pred.injury_probability, "readiness_score": pred.readiness_score,
            "overtraining": bool(pred.overtraining), "last_session": last,
        })
    watch.sort(key=lambda w: (-{"high": 2, "moderate": 1, "low": 0}[w["injury_risk"]], -w["injury_probability"]))

    recent_rows = db.execute(
        select(models.TrainingSession, models.Athlete)
        .join(models.Athlete)
        .options(selectinload(models.TrainingSession.prediction))
        .order_by(models.TrainingSession.session_date.desc(), models.TrainingSession.id.desc())
        .limit(8)
    ).all()
    recent = [{
        "session_id": s.id, "athlete_id": a.id, "name": a.name, "sport": a.sport, "date": s.session_date,
        "duration_min": s.duration_min, "rpe": s.rpe,
        "readiness_score": s.prediction.readiness_score if s.prediction else None,
        "injury_risk": s.prediction.injury_risk if s.prediction else None,
    } for s, a in recent_rows]

    return {
        "athlete_count": len(athletes),
        "session_count": db.query(models.TrainingSession).count(),
        "high_risk": bands["high"],
        "moderate_risk": bands["moderate"],
        "avg_readiness": round(sum(readiness) / len(readiness), 1) if readiness else 0.0,
        "sports": [{"sport": k, "count": v} for k, v in sorted(by_sport.items(), key=lambda x: -x[1])],
        "risk_bands": bands,
        "recent": recent,
        "watchlist": watch[:6],
    }


# --------------------------------------------------------------------------- athletes
@app.get("/api/athletes", response_model=list[schemas.AthleteOut], tags=["athletes"])
def list_athletes(
    sport: schemas.Sport | None = None,
    risk: str | None = Query(default=None, pattern="^(low|moderate|high)$"),
    q: str | None = Query(default=None, max_length=80),
    db: Session = Depends(get_db),
):
    stmt = select(models.Athlete).order_by(models.Athlete.name)
    if sport:
        stmt = stmt.where(models.Athlete.sport == sport)
    if q:
        stmt = stmt.where(models.Athlete.name.ilike(f"%{q.strip()}%"))
    latest, counts = latest_predictions(db), session_counts(db)
    out = [athlete_out(a, latest, counts) for a in db.scalars(stmt).all()]
    if risk:
        out = [a for a in out if a["latest_risk"] == risk]
    return out


@app.post("/api/athletes", response_model=schemas.AthleteOut, status_code=201, tags=["athletes"])
def create_athlete(body: schemas.AthleteCreate, db: Session = Depends(get_db)):
    athlete = models.Athlete(**body.model_dump())
    db.add(athlete)
    db.commit()
    db.refresh(athlete)
    return athlete_out(athlete, {}, {})


@app.get("/api/athletes/{athlete_id}", response_model=schemas.AthleteDetail, tags=["athletes"])
def get_athlete(athlete_id: int, db: Session = Depends(get_db)):
    athlete = _get_athlete(db, athlete_id)
    sessions = sorted(athlete.sessions, key=lambda s: (s.session_date, s.id))
    latest_analysis = engine_singleton.analyze(payload_for(athlete, sessions[-1])) if sessions else None
    return {
        "athlete": athlete_out(athlete, latest_predictions(db), session_counts(db)),
        "sessions": sessions,
        "trend": trend(athlete),
        "latest": latest_analysis,
    }


@app.patch("/api/athletes/{athlete_id}", response_model=schemas.AthleteOut, tags=["athletes"])
def update_athlete(athlete_id: int, body: schemas.AthleteUpdate, db: Session = Depends(get_db)):
    athlete = _get_athlete(db, athlete_id)
    changes = body.model_dump(exclude_unset=True, exclude_none=True)
    for k, v in changes.items():
        setattr(athlete, k, v)
    if any(k in PROFILE_FIELDS for k in changes):  # profile drives the models: rescore history
        for s in athlete.sessions:
            if s.prediction is not None:
                db.delete(s.prediction)
        db.flush()
        for s in athlete.sessions:
            score_session(db, athlete, s)
    db.commit()
    db.refresh(athlete)
    return athlete_out(athlete, latest_predictions(db), session_counts(db))


@app.delete("/api/athletes/{athlete_id}", status_code=204, tags=["athletes"])
def delete_athlete(athlete_id: int, db: Session = Depends(get_db)):
    db.delete(_get_athlete(db, athlete_id))
    db.commit()


# --------------------------------------------------------------------------- sessions
@app.get("/api/athletes/{athlete_id}/sessions", response_model=list[schemas.SessionOut], tags=["sessions"])
def list_sessions(athlete_id: int, db: Session = Depends(get_db)):
    return sorted(_get_athlete(db, athlete_id).sessions, key=lambda s: (s.session_date, s.id))


@app.post("/api/sessions", response_model=schemas.SessionLogged, status_code=201, tags=["sessions"])
def create_session(body: schemas.SessionCreate, db: Session = Depends(get_db)):
    athlete = _get_athlete(db, body.athlete_id)
    sess = models.TrainingSession(**body.model_dump())
    db.add(sess)
    db.flush()
    analysis = score_session(db, athlete, sess)
    db.commit()
    db.refresh(sess)
    return {"session": sess, "analysis": analysis}


@app.delete("/api/sessions/{session_id}", status_code=204, tags=["sessions"])
def delete_session(session_id: int, db: Session = Depends(get_db)):
    sess = db.get(models.TrainingSession, session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    db.delete(sess)
    db.commit()


# --------------------------------------------------------------------------- analysis
@app.post("/api/analyze", response_model=schemas.AnalysisResult, tags=["analysis"])
def analyze(body: schemas.AnalyzePayload, db: Session = Depends(get_db)):
    payload = body.model_dump(exclude={"athlete_id"})
    if body.athlete_id is not None:
        # What-if: start from the athlete's profile + latest session, override with explicit fields.
        athlete = _get_athlete(db, body.athlete_id)
        base = {k: getattr(athlete, k) for k in PROFILE_FIELDS}
        if athlete.sessions:
            last = max(athlete.sessions, key=lambda s: (s.session_date, s.id))
            base.update({k: getattr(last, k) for k in SESSION_FIELDS})
        explicit = body.model_fields_set - {"athlete_id"}
        payload = {**payload, **base, **{k: payload[k] for k in explicit}}
    return engine_singleton.analyze(payload)


# --------------------------------------------------------------------------- CSV ingest
REQUIRED_CSV = {"name", "sport", "age", "height_cm", "weight_kg", "session_date", "duration_min"}


@app.post("/api/upload-csv", response_model=schemas.UploadResult, tags=["ingest"])
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")
    raw = await file.read(settings.max_upload_bytes + 1)
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"CSV larger than {settings.max_upload_bytes // 1000} KB")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded")
    reader = csv.DictReader(io.StringIO(text))
    headers = {h.strip() for h in (reader.fieldnames or []) if h}
    missing = sorted(REQUIRED_CSV - headers)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing)}")

    by_key = {(a.name.lower(), a.sport): a for a in db.scalars(select(models.Athlete)).all()}
    rows = created_a = created_s = 0
    errors: list[dict] = []
    for line_no, raw_row in enumerate(reader, start=2):
        rows += 1
        if rows > MAX_CSV_ROWS:
            errors.append({"line": line_no, "error": f"Row limit of {MAX_CSV_ROWS} reached; remaining rows skipped"})
            break
        row = {k.strip(): (v or "").strip() for k, v in raw_row.items() if k}
        row = {k: v for k, v in row.items() if v != ""}
        try:
            profile = schemas.AthleteCreate(**{k: v for k, v in row.items() if k in schemas.AthleteCreate.model_fields})
            session_in = schemas.SessionMetrics(**{k: v for k, v in row.items() if k in schemas.SessionMetrics.model_fields})
            session_date = date.fromisoformat(row["session_date"])
            if session_date > date.today():
                raise ValueError("session_date cannot be in the future")
        except ValidationError as exc:
            e = exc.errors()[0]
            errors.append({"line": line_no, "error": f"{'.'.join(map(str, e['loc']))}: {e['msg']}"})
            continue
        except ValueError as exc:
            errors.append({"line": line_no, "error": f"session_date: {exc}"})
            continue
        key = (profile.name.lower(), profile.sport)
        athlete = by_key.get(key)
        if athlete is None:
            athlete = models.Athlete(**profile.model_dump())
            db.add(athlete)
            db.flush()
            by_key[key] = athlete
            created_a += 1
        sess = models.TrainingSession(athlete_id=athlete.id, session_date=session_date,
                                      notes=row.get("notes", "CSV import")[:1000], **session_in.model_dump())
        db.add(sess)
        db.flush()
        score_session(db, athlete, sess)
        created_s += 1
    db.commit()
    return {"rows": rows, "athletes_created": created_a, "sessions_created": created_s, "errors": errors[:50]}


# --------------------------------------------------------------------------- static UI
@app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
def api_not_found(rest: str):
    raise HTTPException(status_code=404, detail=f"Unknown API route /api/{rest}")


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str):
    if path:
        candidate = (DIST / path).resolve()
        if candidate.is_file() and DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
    index = DIST / "index.html"
    if not index.exists():
        return JSONResponse({"app": settings.app_name, "docs": "/docs"})
    return FileResponse(index, headers={"Cache-Control": "no-cache"})
