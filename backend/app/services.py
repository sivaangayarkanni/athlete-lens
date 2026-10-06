"""Domain helpers shared by the API routes, CSV ingest and the seeder."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import models
from .injury.catalog import REGIONS
from .injury.engine import daily_from_sessions, engine as injury_engine, injury_events
from .injury.features import FIDX, daily_features
from .ml_engine import DEFAULT_JUMP, DEFAULT_SPRINT, engine_singleton

ACWR_MIN_HISTORY_DAYS = 21
MIN_CALENDAR_DAYS = 28
PROFILE_FIELDS = ("age", "sex", "sport", "years_training", "height_cm", "weight_kg", "previous_injuries")
SESSION_FIELDS = (
    "duration_min", "distance_km", "sprint_100m_s", "vertical_jump_cm", "resting_hr", "session_hr_avg",
    "rpe", "sleep_hours", "wellness", "sessions_last_7", "rest_days_last_7",
)


# ---------------------------------------------------------------------------- profile + calendar
def injury_profile(athlete: models.Athlete) -> dict:
    return {"age": athlete.age, "sex": athlete.sex, "sport": athlete.sport, "height_cm": athlete.height_cm,
            "weight_kg": athlete.weight_kg, "years_training": athlete.years_training,
            "growth_cm": athlete.growth_cm or 0.0, "nordic": int(bool(athlete.nordic_program)),
            "adductor": int(bool(athlete.adductor_program))}


def calendar(athlete: models.Athlete, as_of: date | None = None):
    """Daily arrays from the first session to `as_of` (default: last session).

    If fewer than 28 days are logged, the first logged week is replicated backwards so chronic
    load is not under-estimated (flagged as `padded`).
    """
    sessions = sorted(athlete.sessions, key=lambda s: (s.session_date, s.id))
    if not sessions:
        return None
    end = as_of or sessions[-1].session_date
    sessions = [s for s in sessions if s.session_date <= end]
    if not sessions:
        return None
    first = sessions[0].session_date
    start = min(first, end - timedelta(days=MIN_CALENDAR_DAYS - 1))
    daily = daily_from_sessions(sessions, start, end)
    pad = (first - start).days
    if pad > 0:   # replicate the first logged week backwards
        src = daily_from_sessions(sessions, first, min(end, first + timedelta(days=6)))
        L = len(src["load"])
        for k in daily:
            for i in range(pad):
                daily[k][pad - 1 - i] = src[k][(L - 1 - (i % L))]
    events, prior = injury_events(athlete.injuries, start, athlete.injury_history)
    return {"daily": daily, "start": start, "end": end, "events": events, "prior": prior, "padded": pad > 0}


def load_snapshot(X_row: np.ndarray) -> dict:
    g = lambda f: None if np.isnan(X_row[FIDX[f]]) else round(float(X_row[FIDX[f]]), 2)  # noqa: E731
    return {"acute_7": g("acute_7"), "chronic_28w": g("chronic_28w"), "acwr": g("acwr_ewma"), "acwr_rolling": g("acwr_ra"),
            "monotony": g("monotony_7"), "strain": g("strain_7"), "load_change_wow": g("load_wow"),
            "sleep_debt_7": g("sleep_debt_7"), "matches_7": g("matches_7"), "sessions_7": g("sessions_7")}


def injury_summary(athlete: models.Athlete, as_of: date | None = None, explain: str | None = None,
                   counterfactual: bool = False, cal: dict | None = None) -> dict | None:
    cal = cal or calendar(athlete, as_of)
    if cal is None:
        return None
    profile = injury_profile(athlete)
    X = daily_features(cal["daily"], profile, cal["events"], cal["prior"])
    x = X[-1]
    out = injury_engine.summarize(x, athlete.sport)
    out["as_of"] = cal["end"]
    out["load"] = load_snapshot(x)
    out["load"]["history_padded"] = cal["padded"]
    out["real_data"] = injury_engine.real_check(cal["daily"])
    if explain:
        targets = ["any"] + (REGIONS if explain == "all" else out["top_regions"])
        out["drivers"] = injury_engine.explain(x, athlete.sport, targets)
    if counterfactual:
        out["counterfactuals"] = injury_engine.counterfactuals(cal["daily"], profile, cal["events"], cal["prior"],
                                                               ["any"] + out["top_regions"])
    return out


# ---------------------------------------------------------------------------- legacy payload
def last_known(athlete: models.Athlete, attr: str, upto: models.TrainingSession | None = None):
    for s in sorted(athlete.sessions, key=lambda s: (s.session_date, s.id), reverse=True):
        if upto is not None and (s.session_date, s.id or 0) > (upto.session_date, upto.id or 0):
            continue
        v = getattr(s, attr)
        if v is not None:
            return v
    return None


def payload_for(athlete: models.Athlete, sess) -> dict:
    payload = {k: getattr(athlete, k) for k in PROFILE_FIELDS}
    for k in SESSION_FIELDS:
        payload[k] = sess[k] if isinstance(sess, dict) else getattr(sess, k)
    sex = "F" if athlete.sex == "F" else "M"
    if payload.get("sprint_100m_s") is None:
        payload["sprint_100m_s"] = last_known(athlete, "sprint_100m_s") or DEFAULT_SPRINT[sex]
    if payload.get("vertical_jump_cm") is None:
        payload["vertical_jump_cm"] = last_known(athlete, "vertical_jump_cm") or DEFAULT_JUMP[sex]
    return payload


def _store(db: Session, athlete: models.Athlete, sess: models.TrainingSession, result: dict, region_p: dict):
    top = max(region_p, key=region_p.get) if region_p else None
    db.add(models.PredictionLog(
        athlete_id=athlete.id, session_id=sess.id, performance_index=result["performance_index"],
        readiness_score=result["readiness_score"], injury_risk=result["injury_risk"],
        injury_probability=result["injury_probability"], overtraining=int(result["overtraining"]),
        region_risks=json.dumps({r: round(float(v), 5) for r, v in region_p.items()}), top_region=top,
    ))


def score_session(db: Session, athlete: models.Athlete, sess: models.TrainingSession) -> dict:
    """Score one stored session (injury model as of that day) and persist the prediction."""
    summary = injury_summary(athlete, as_of=sess.session_date, explain="top")
    result = engine_singleton.analyze(payload_for(athlete, sess), summary)
    region_p = {r["region"]: r["probability"] for r in summary["regions"]} if summary else {}
    _store(db, athlete, sess, result, region_p)
    return result


def rescore_athlete(db: Session, athlete: models.Athlete) -> None:
    """Batch-score every session of an athlete (one feature pass, one predict call per model)."""
    for s in athlete.sessions:
        if s.prediction is not None:
            db.delete(s.prediction)
    db.flush()
    db.expire(athlete, ["sessions"])
    sessions = sorted(athlete.sessions, key=lambda s: (s.session_date, s.id))
    if not sessions:
        return
    cal = calendar(athlete)
    X = daily_features(cal["daily"], injury_profile(athlete), cal["events"], cal["prior"])
    idx = [(s.session_date - cal["start"]).days for s in sessions]
    P = injury_engine.predict(X[idx])
    for k, s in enumerate(sessions):
        p_any = float(P["any"][k])
        summary = {"overall": {"probability": round(p_any, 4), "band": injury_engine.band(p_any)}}
        result = engine_singleton.analyze(payload_for(athlete, s), None)
        result["injury_probability"], result["injury_risk"] = summary["overall"]["probability"], summary["overall"]["band"]
        _store(db, athlete, s, result, {r: float(P[r][k]) for r in REGIONS})


# ---------------------------------------------------------------------------- read models
def latest_predictions(db: Session) -> dict[int, tuple[models.PredictionLog, date]]:
    rows = db.execute(
        select(models.PredictionLog, models.TrainingSession.session_date)
        .join(models.TrainingSession, models.TrainingSession.id == models.PredictionLog.session_id)
        .order_by(models.TrainingSession.session_date, models.TrainingSession.id)
    ).all()
    out: dict[int, tuple[models.PredictionLog, date]] = {}
    for pred, d in rows:
        out[pred.athlete_id] = (pred, d)
    return out


def athlete_out(athlete: models.Athlete, latest: dict, counts: dict) -> dict:
    data = {c.name: getattr(athlete, c.name) for c in models.Athlete.__table__.columns}
    data["nordic_program"] = bool(athlete.nordic_program)
    data["adductor_program"] = bool(athlete.adductor_program)
    pred, last = latest.get(athlete.id, (None, None))
    data.update(
        latest_readiness=pred.readiness_score if pred else None,
        latest_performance=pred.performance_index if pred else None,
        latest_risk=pred.injury_risk if pred else None,
        latest_injury_probability=pred.injury_probability if pred else None,
        top_region=pred.top_region if pred else None,
        last_session=last,
        session_count=counts.get(athlete.id, 0),
    )
    return data


def session_counts(db: Session) -> dict[int, int]:
    return dict(db.execute(
        select(models.TrainingSession.athlete_id, func.count()).group_by(models.TrainingSession.athlete_id)
    ).all())


def trend(athlete: models.Athlete) -> list[dict]:
    """Per-session timeline: session-RPE load, rolling acute/chronic, ACWR and stored predictions."""
    sessions = sorted(athlete.sessions, key=lambda s: (s.session_date, s.id))
    daily: dict[date, float] = defaultdict(float)
    for s in sessions:
        daily[s.session_date] += s.duration_min * s.rpe

    def window(end: date, days: int) -> float:
        return sum(v for d, v in daily.items() if end - timedelta(days=days - 1) <= d <= end)

    first = sessions[0].session_date if sessions else None
    points = []
    for s in sessions:
        acute = window(s.session_date, 7)
        chronic = window(s.session_date, 28) / 4
        pred = s.prediction
        points.append({
            "date": s.session_date, "session_id": s.id, "srpe_load": round(s.duration_min * s.rpe, 1),
            "acute_7d": round(acute, 1), "chronic_28d": round(chronic, 1),
            "acwr": round(acute / chronic, 2) if chronic > 0 and (s.session_date - first).days >= ACWR_MIN_HISTORY_DAYS else None,
            "readiness_score": pred.readiness_score if pred else None,
            "performance_index": pred.performance_index if pred else None,
            "injury_probability": pred.injury_probability if pred else None,
            "injury_risk": pred.injury_risk if pred else None,
            "top_region": pred.top_region if pred else None,
            "sprint_100m_s": s.sprint_100m_s, "vertical_jump_cm": s.vertical_jump_cm,
            "sleep_hours": s.sleep_hours, "wellness": s.wellness, "session_type": s.session_type,
        })
    return points


def daily_risk_trend(athlete: models.Athlete) -> dict | None:
    """Day-by-day model risk (all regions) + load-management metrics over the logged calendar."""
    cal = calendar(athlete)
    if cal is None:
        return None
    X = daily_features(cal["daily"], injury_profile(athlete), cal["events"], cal["prior"])
    first_logged = min(s.session_date for s in athlete.sessions)
    keep = [i for i in range((first_logged - cal["start"]).days, len(X))]
    P = injury_engine.predict(X[keep])
    col = lambda f: [None if np.isnan(v) else round(float(v), 3) for v in X[keep, FIDX[f]]]  # noqa: E731
    return {
        "dates": [cal["start"] + timedelta(days=i) for i in keep],
        "overall": [round(float(v), 4) for v in P["any"]],
        "regions": {r: [round(float(v), 5) for v in P[r]] for r in REGIONS},
        "load": {"daily_load": [round(float(cal["daily"]["load"][i]), 1) for i in keep],
                 "ewma_acute": col("ewma_acute"), "ewma_chronic": col("ewma_chronic"), "acwr": col("acwr_ewma"),
                 "monotony": col("monotony_7"), "strain": col("strain_7"), "sleep_debt_7": col("sleep_debt_7"),
                 "match": [int(cal["daily"]["match"][i]) for i in keep]},
        "bands": injury_engine.m["bands"],
    }


# ---------------------------------------------------------------------------- what-if / guest analysis
def _week_days(n: int) -> list[int]:
    """Spread n sessions over a 7-day week (indices 0-6), always including the last day."""
    n = int(min(max(n, 0), 7))
    if n == 0:
        return []
    return sorted({6 - int(round(i * 7 / n)) for i in range(n)})


def guest_daily(p: dict) -> dict:
    """Synthetic 4-week calendar from a single-session form (guest mode).

    Weeks 1-3 use the 'typical' sessions/week and duration; the last 7 days use sessions_last_7
    with today's session on the final day. Assumption-driven: shown as such in the UI.
    """
    from .injury.engine import empty_daily
    d = empty_daily(28)
    chronic_n = p.get("chronic_sessions_per_week") if p.get("chronic_sessions_per_week") is not None else p["sessions_last_7"]
    chronic_dur = p.get("chronic_duration_min") or p["duration_min"]
    chronic_rpe = max(1.0, p["rpe"] - 0.5)
    dist_per_min = (p.get("distance_km") or 0.0) / max(p["duration_min"], 1)

    def put(i, dur, rpe, match=0):
        d["load"][i] += dur * rpe
        d["duration"][i] += dur
        d["distance"][i] += dur * dist_per_min
        d["rpe"][i] = rpe
        d["session"][i] += 1
        d["match"][i] = max(d["match"][i], match)

    for w in range(3):
        n = int(chronic_n)
        days = _week_days(n)
        for j in days:
            put(w * 7 + j, chronic_dur * n / max(len(days), 1) * (0.9 + 0.1 * (j % 3)), chronic_rpe)
    n7 = int(p["sessions_last_7"])
    days = _week_days(max(n7, 1))
    matches = min(int(p.get("matches_last_7") or 0), len(days))
    # matches go on the most recent sessions before today; today's session too if there are not enough days
    before = days[:-1]
    match_days = set(before[len(before) - min(matches, len(before)):]) if matches else set()
    if matches > len(before):
        match_days.add(days[-1])
    for j in days:
        last = j == 6
        put(21 + j, p["duration_min"] * (n7 / max(len(days), 1) if n7 > 7 else 1.0) * (1.0 if last else 0.95),
            p["rpe"] if last or j in match_days else max(1.0, p["rpe"] - 0.5), 1.0 if j in match_days else 0.0)
    d["sleep"][:21] = max(p["sleep_hours"], 6.0) if p["sleep_hours"] >= 7 else p["sleep_hours"] + 0.3
    d["sleep"][21:] = p["sleep_hours"]
    d["wellness"][:] = p["wellness"]
    d["rhr"][:] = p["resting_hr"]
    d["sprint"][27] = p["sprint_100m_s"]
    d["jump"][27] = p["vertical_jump_cm"]
    if p.get("asymmetry_pct") is not None:
        d["asymmetry"][27] = p["asymmetry_pct"]
    return d


def apply_what_if(daily: dict, load_change_pct: float = 0.0, extra_rest_days: int = 0, sleep_hours: float | None = None) -> dict:
    d = {k: v.copy() for k, v in daily.items()}
    last = slice(max(0, len(d["load"]) - 7), len(d["load"]))
    if load_change_pct:
        f = 1 + load_change_pct / 100
        for k in ("load", "duration", "distance"):
            d[k][last] *= f
    for _ in range(int(extra_rest_days)):
        loads = np.where((d["session"][last] > 0) & (d["match"][last] == 0), d["load"][last], -1)
        loads[-1] = -1   # keep today's session
        if loads.max() <= 0:
            break
        i = last.start + int(loads.argmax())
        for k in ("load", "duration", "distance", "session"):
            d[k][i] = 0.0
        d["rpe"][i] = np.nan
    if sleep_hours is not None:
        d["sleep"][last] = sleep_hours
    return d


def analysis_injury(payload: dict, athlete: models.Athlete | None, explicit: set[str]) -> dict:
    """Per-region injury summary for /api/analyze (guest form or athlete what-if)."""
    if athlete is not None and athlete.sessions:
        cal = calendar(athlete)
        profile = injury_profile(athlete)
        for k, pk in (("nordic_program", "nordic"), ("adductor_program", "adductor"), ("growth_cm", "growth_cm")):
            if k in explicit and payload.get(k) is not None:
                profile[pk] = payload[k] if k == "growth_cm" else int(bool(payload[k]))
        last = max(athlete.sessions, key=lambda s: (s.session_date, s.id))
        sleep = payload["sleep_hours"] if "sleep_hours" in explicit and abs(payload["sleep_hours"] - last.sleep_hours) > 1e-6 else None
        daily = apply_what_if(cal["daily"], payload.get("load_change_pct") or 0, payload.get("extra_rest_days") or 0, sleep)
        events, prior, sport, as_of, padded = cal["events"], cal["prior"], athlete.sport, cal["end"], cal["padded"]
    else:
        profile = {k: payload[k] for k in ("age", "sex", "sport", "height_cm", "weight_kg", "years_training")}
        profile.update(growth_cm=payload.get("growth_cm") or 0.0, nordic=int(bool(payload.get("nordic_program"))),
                       adductor=int(bool(payload.get("adductor_program"))))
        daily = apply_what_if(guest_daily(payload), payload.get("load_change_pct") or 0, payload.get("extra_rest_days") or 0)
        events, prior, sport, as_of, padded = [], {}, payload["sport"], date.today(), True
        for r in payload.get("injury_history") or []:
            prior[r] = prior.get(r, 0) + 1
    X = daily_features(daily, profile, events, prior)
    x = X[-1]
    out = injury_engine.summarize(x, sport)
    out["as_of"] = as_of
    out["load"] = load_snapshot(x)
    out["load"]["history_padded"] = padded
    out["load"]["synthetic_history"] = athlete is None or not athlete.sessions
    out["drivers"] = injury_engine.explain(x, sport, ["any"] + out["top_regions"])
    out["real_data"] = injury_engine.real_check(daily)
    return out
