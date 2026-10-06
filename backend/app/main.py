"""Athlete Lens API + static production UI."""
from __future__ import annotations

import csv
import io
import json
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
from .database import Base, SessionLocal, engine, migrate
from .injury.catalog import INJURY_TYPES, PREVENTION, REGION_LABELS, REGIONS, SOURCES, TYPE_LABELS
from .injury.engine import engine as injury_engine, prevention_for
from .injury.features import FEATURES as INJURY_FEATURES
from .ml_engine import SPORTS, engine_singleton
from .seed import seed_if_empty
from .services import (
    PROFILE_FIELDS,
    SESSION_FIELDS,
    analysis_injury,
    athlete_out,
    daily_risk_trend,
    injury_summary,
    latest_predictions,
    payload_for,
    rescore_athlete,
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
        added = migrate()
        if added:
            log.info("migrated columns: %s", ", ".join(added))
        engine_singleton.load()
        injury_engine.load()
        if settings.seed_demo_data:
            with SessionLocal() as db:
                seed_if_empty(db)
        _ready = True
        log.info("Athlete Lens ready (performance model from %s, injury models from %s)",
                 engine_singleton.source, injury_engine.path)


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


def _athlete_fields(data: dict) -> dict:
    """Pydantic -> ORM column values (region list stored as CSV text, booleans as ints)."""
    out = dict(data)
    if "injury_history" in out and out["injury_history"] is not None:
        out["injury_history"] = ",".join(out["injury_history"])
    for k in ("nordic_program", "adductor_program"):
        if k in out and out[k] is not None:
            out[k] = int(bool(out[k]))
    return out


RESCORE_FIELDS = set(PROFILE_FIELDS) | {"injury_history", "growth_cm", "nordic_program", "adductor_program"}


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
    im = injury_engine.metrics.get("simulated", {}).get("grouped_cv", {}).get("any", {}).get("model", {})
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
            "injury_ready": injury_engine.ready,
            "injury_regions": len(REGIONS),
            "injury_any_grouped_cv_roc_auc": im.get("roc_auc"),
            "performance_holdout_mae": m.get("performance", {}).get("holdout_mae"),
        },
    }


@app.get("/api/sports", tags=["meta"])
def sports():
    return SPORTS


INJURY_DESIGN = (
    "13 calibrated HistGradientBoosting classifiers (any injury + 12 body regions) predicting injury onset in the next "
    f"7 days from {len(INJURY_FEATURES)} daily features: EWMA acute/chronic load, ACWR, week-over-week spikes, Foster monotony and strain, "
    "sleep debt, wellness trend, sprint/jump/resting-HR drift, asymmetry, injury history per region, growth, prevention "
    "programmes, sport and the sport's typical region mix. Trees are shallow and heavily regularised because rows from "
    "one athlete are strongly correlated. Probabilities are calibrated with CalibratedClassifierCV (sigmoid / Platt "
    "scaling) on athlete-grouped folds. The likely injury type per region comes from a regularised HGB multiclass model, "
    "or from the region's type-frequency table when the model does not beat it in grouped CV, and a model trained on real runner logs (Lövdal et al. 2021, CC0) is shown as an independent cross-check."
)

LIMITATIONS = [
    "The per-region models are trained on SIMULATED athlete-seasons. The simulator's injury rates and region shares "
    "follow published epidemiology where we could read it (see sources), but how much each factor matters is partly "
    "assumed. Metrics show how well the model recovers the simulator, not real-world accuracy.",
    "The only real data is Lövdal et al. (2021): 74 Dutch competitive runners, binary injury labels with no body "
    "region. It powers the separate real-data cross-check, whose held-out AUC is only modestly above chance.",
    "Injury prediction is hard even with real data; treat probabilities as prompts for a conversation and a load "
    "review, not a forecast.",
    "Not a medical device. Prevention and return-to-play notes are general guidance and do not replace a physiotherapist or doctor.",
    "Before relying on probabilities, retrain on at least one season of real, region-labelled squad data.",
]


@app.get("/api/model", tags=["meta"])
def model_card():
    engine_singleton.load()
    injury_engine.load()
    meta = engine_singleton.meta
    im = injury_engine.metrics
    any_cv = im.get("simulated", {}).get("grouped_cv", {}).get("any", {})
    return {
        "name": "Athlete Lens v2",
        "performance_model": meta.get("performance_model"),
        "injury_model": INJURY_DESIGN,
        "training_data": {
            "performance": {"kind": "synthetic", "cohort_size": meta.get("cohort_size"), "train_rows": meta.get("train_rows"),
                            "test_rows": meta.get("test_rows"), "seed": meta.get("seed")},
            "injury": {"kind": "simulated athlete-seasons + real runner logs (cross-check)",
                       "simulated": {k: im.get("simulated", {}).get(k) for k in ("n_athletes", "season_days", "rows", "injuries", "injuries_per_athlete_season")},
                       "real": im.get("real", {}).get("source")},
        },
        "metrics": {
            "performance": meta.get("metrics", {}).get("performance"),
            "injury_any_grouped_cv": any_cv,
        },
        "trained_at": meta.get("trained_at"),
        "injury_trained_at": im.get("trained_at"),
        "sklearn_version": meta.get("sklearn_version"),
        "features": meta.get("features", []),
        "injury_features": len(im.get("features", [])),
        "risk_bands": {
            "high": f"7-day any-injury probability >= {injury_engine.m['bands']['high']:.3f} (top ~15% of simulated athlete-days)",
            "moderate": f">= {injury_engine.m['bands']['moderate']:.3f} (top ~40%)",
            "low": "otherwise",
            "region_levels": "relative to the sport average: high >= 2.0x, elevated >= 1.3x, typical >= 0.7x, low < 0.7x",
        },
        "readiness_bands": {"green": ">= 65", "amber": "45-64", "red": "< 45"},
        "intended_use": "Coach-facing load and recovery triage for school, college and district athletes "
                        "without lab equipment.",
        "limitations": LIMITATIONS,
        "injury_model_details": "/api/injury-model",
    }


@app.get("/api/injury-model", tags=["injury"])
def injury_model_card():
    """Full injury-model card: design, every metric from the training run, sources and limitations."""
    injury_engine.load()
    return {"design": INJURY_DESIGN, **injury_engine.metrics, "sources": SOURCES, "limitations": LIMITATIONS,
            "bands": injury_engine.m["bands"]}


@app.get("/api/regions", tags=["injury"])
def regions():
    return [{"region": r, "label": REGION_LABELS[r], **PREVENTION[r]} for r in REGIONS]


@app.get("/api/injury-types", tags=["injury"])
def injury_types():
    return [{"type": t, "label": TYPE_LABELS[t]} for t in INJURY_TYPES]


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
            "top_region": pred.top_region, "top_region_label": REGION_LABELS.get(pred.top_region or "", None),
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
    athlete = models.Athlete(**_athlete_fields(body.model_dump()))
    db.add(athlete)
    db.commit()
    db.refresh(athlete)
    return athlete_out(athlete, {}, {})


@app.get("/api/athletes/{athlete_id}", response_model=schemas.AthleteDetail, tags=["athletes"])
def get_athlete(athlete_id: int, db: Session = Depends(get_db)):
    athlete = _get_athlete(db, athlete_id)
    sessions = sorted(athlete.sessions, key=lambda s: (s.session_date, s.id))
    latest_analysis = (engine_singleton.analyze(payload_for(athlete, sessions[-1]), injury_summary(athlete, explain="top"))
                       if sessions else None)
    return {
        "athlete": athlete_out(athlete, latest_predictions(db), session_counts(db)),
        "sessions": sessions,
        "trend": trend(athlete),
        "latest": latest_analysis,
    }


@app.patch("/api/athletes/{athlete_id}", response_model=schemas.AthleteOut, tags=["athletes"])
def update_athlete(athlete_id: int, body: schemas.AthleteUpdate, db: Session = Depends(get_db)):
    athlete = _get_athlete(db, athlete_id)
    changes = _athlete_fields(body.model_dump(exclude_unset=True, exclude_none=True))
    for k, v in changes.items():
        setattr(athlete, k, v)
    if RESCORE_FIELDS & set(changes):  # profile drives the models: rescore history
        db.flush()
        rescore_athlete(db, athlete)
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
    athlete = None
    explicit = body.model_fields_set - {"athlete_id"}
    if body.athlete_id is not None:
        # What-if: start from the athlete's profile + latest session, override with explicit fields.
        athlete = _get_athlete(db, body.athlete_id)
        base = {k: getattr(athlete, k) for k in PROFILE_FIELDS}
        if athlete.sessions:
            last = max(athlete.sessions, key=lambda s: (s.session_date, s.id))
            base.update({k: v for k, v in payload_for(athlete, last).items() if k in SESSION_FIELDS})
        payload = {**payload, **base, **{k: payload[k] for k in explicit}}
    injury = analysis_injury(payload, athlete, explicit)
    return engine_singleton.analyze(payload, injury)


# --------------------------------------------------------------------------- injury risk (per region)
def _need_sessions(athlete: models.Athlete) -> None:
    if not athlete.sessions:
        raise HTTPException(status_code=409, detail="Log at least one session before requesting injury risk")


@app.get("/api/athletes/{athlete_id}/injury-risk", response_model=schemas.InjuryRiskSummary, tags=["injury"])
def athlete_injury_risk(athlete_id: int, explain: str = Query(default="top", pattern="^(none|top|all)$"),
                        db: Session = Depends(get_db)):
    """Current 7-day risk for all 12 regions with Shapley-style drivers and the real-data cross-check."""
    athlete = _get_athlete(db, athlete_id)
    _need_sessions(athlete)
    return injury_summary(athlete, explain=None if explain == "none" else explain)


@app.get("/api/athletes/{athlete_id}/injury-risk/{region}", tags=["injury"])
def athlete_region_detail(athlete_id: int, region: schemas.Region, db: Session = Depends(get_db)):
    """One region: probability, drivers, injury-type mix, what-if scenarios, trend and prevention / RTP guidance."""
    athlete = _get_athlete(db, athlete_id)
    _need_sessions(athlete)
    summary = injury_summary(athlete, explain=None)
    from .services import calendar, injury_profile
    from .injury.features import daily_features
    cal = calendar(athlete)
    X = daily_features(cal["daily"], injury_profile(athlete), cal["events"], cal["prior"])
    drivers = injury_engine.explain(X[-1], athlete.sport, [region])[region]
    cfs = injury_engine.counterfactuals(cal["daily"], injury_profile(athlete), cal["events"], cal["prior"], [region, "any"])
    rt = daily_risk_trend(athlete)
    item = next(r for r in summary["regions"] if r["region"] == region)
    history = [{"id": i.id, "injury_type": i.injury_type, "onset_date": i.onset_date, "return_date": i.return_date}
               for i in athlete.injuries if i.region == region]
    prior = [r for r in (athlete.injury_history or "").split(",") if r.strip() == region]
    return {
        "athlete_id": athlete.id, "as_of": summary["as_of"], **item,
        "rank": 1 + sorted(summary["regions"], key=lambda r: -r["relative_risk"]).index(item),
        "drivers": drivers, "counterfactuals": cfs,
        "trend": {"dates": rt["dates"], "probability": rt["regions"][region]},
        "history": history, "previous_injuries_before_tracking": len(prior),
        "guidance": prevention_for(region),
    }


@app.get("/api/athletes/{athlete_id}/risk-trend", tags=["injury"])
def athlete_risk_trend(athlete_id: int, db: Session = Depends(get_db)):
    """Daily overall + per-region risk and load-management metrics (EWMA, ACWR, monotony, strain)."""
    athlete = _get_athlete(db, athlete_id)
    _need_sessions(athlete)
    return daily_risk_trend(athlete)


@app.get("/api/athletes/{athlete_id}/what-if", tags=["injury"])
def athlete_what_if(athlete_id: int, db: Session = Depends(get_db)):
    """Counterfactual scenarios (cut load, rest day, sleep, skip match, prevention programmes) for the top regions."""
    athlete = _get_athlete(db, athlete_id)
    _need_sessions(athlete)
    s = injury_summary(athlete, explain=None, counterfactual=True)
    return {"athlete_id": athlete.id, "as_of": s["as_of"], "top_regions": s["top_regions"], "scenarios": s["counterfactuals"]}


@app.get("/api/squad/heatmap", tags=["injury"])
def squad_heatmap(sport: schemas.Sport | None = None, db: Session = Depends(get_db)):
    """Athletes x regions matrix of the latest stored 7-day probabilities (and relative risk vs sport average)."""
    injury_engine.load()
    latest = latest_predictions(db)
    rows = []
    stmt = select(models.Athlete).order_by(models.Athlete.name)
    if sport:
        stmt = stmt.where(models.Athlete.sport == sport)
    for a in db.scalars(stmt).all():
        pred, last = latest.get(a.id, (None, None))
        if pred is None or not pred.region_risks:
            continue
        risks = json.loads(pred.region_risks)
        means = injury_engine.m["sport_mean"].get(a.sport, injury_engine.m["sport_mean"]["Athletics"])
        rows.append({
            "athlete_id": a.id, "name": a.name, "sport": a.sport, "as_of": last,
            "overall": pred.injury_probability, "band": pred.injury_risk, "top_region": pred.top_region,
            "probabilities": {r: risks.get(r) for r in REGIONS},
            "relative_risk": {r: round(risks.get(r, 0) / max(means[r], 1e-6), 2) for r in REGIONS},
        })
    rows.sort(key=lambda r: -r["overall"])
    return {"regions": [{"region": r, "label": REGION_LABELS[r]} for r in REGIONS], "athletes": rows}


@app.get("/api/athletes/{athlete_id}/injuries", response_model=list[schemas.InjuryOut], tags=["injury"])
def list_injuries(athlete_id: int, db: Session = Depends(get_db)):
    return sorted(_get_athlete(db, athlete_id).injuries, key=lambda i: i.onset_date)


@app.post("/api/athletes/{athlete_id}/injuries", response_model=schemas.InjuryOut, status_code=201, tags=["injury"])
def create_injury(athlete_id: int, body: schemas.InjuryCreate, db: Session = Depends(get_db)):
    athlete = _get_athlete(db, athlete_id)
    if body.return_date and body.return_date < body.onset_date:
        raise HTTPException(status_code=422, detail="return_date must be on or after onset_date")
    rec = models.InjuryRecord(athlete_id=athlete.id, **body.model_dump())
    db.add(rec)
    athlete.previous_injuries = min(20, (athlete.previous_injuries or 0) + 1)
    db.flush()
    db.refresh(athlete)
    rescore_athlete(db, athlete)
    db.commit()
    db.refresh(rec)
    return rec


@app.delete("/api/injuries/{injury_id}", status_code=204, tags=["injury"])
def delete_injury(injury_id: int, db: Session = Depends(get_db)):
    rec = db.get(models.InjuryRecord, injury_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Injury {injury_id} not found")
    athlete = rec.athlete
    db.delete(rec)
    athlete.previous_injuries = max(0, (athlete.previous_injuries or 0) - 1)
    db.flush()
    db.refresh(athlete)
    rescore_athlete(db, athlete)
    db.commit()


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
    touched: dict[int, models.Athlete] = {}
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
            athlete = models.Athlete(**_athlete_fields(profile.model_dump()))
            db.add(athlete)
            db.flush()
            by_key[key] = athlete
            created_a += 1
        sess = models.TrainingSession(athlete_id=athlete.id, session_date=session_date,
                                      notes=row.get("notes", "CSV import")[:1000], **session_in.model_dump())
        db.add(sess)
        db.flush()
        touched[athlete.id] = athlete
        created_s += 1
    for athlete in touched.values():   # batch-score each athlete's full history once
        db.refresh(athlete)
        rescore_athlete(db, athlete)
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
