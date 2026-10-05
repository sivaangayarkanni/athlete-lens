"""Domain helpers shared by the API routes, CSV ingest and the seeder."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import models
from .ml_engine import engine_singleton

ACWR_MIN_HISTORY_DAYS = 21
PROFILE_FIELDS = ("age", "sex", "sport", "years_training", "height_cm", "weight_kg", "previous_injuries")
SESSION_FIELDS = (
    "duration_min", "distance_km", "sprint_100m_s", "vertical_jump_cm", "resting_hr", "session_hr_avg",
    "rpe", "sleep_hours", "wellness", "sessions_last_7", "rest_days_last_7",
)


def payload_for(athlete: models.Athlete, sess: models.TrainingSession | dict) -> dict:
    payload = {k: getattr(athlete, k) for k in PROFILE_FIELDS}
    for k in SESSION_FIELDS:
        payload[k] = sess[k] if isinstance(sess, dict) else getattr(sess, k)
    return payload


def score_session(db: Session, athlete: models.Athlete, sess: models.TrainingSession) -> dict:
    """Run the models on a stored session and persist the prediction."""
    result = engine_singleton.analyze(payload_for(athlete, sess))
    db.add(models.PredictionLog(
        athlete_id=athlete.id,
        session_id=sess.id,
        performance_index=result["performance_index"],
        readiness_score=result["readiness_score"],
        injury_risk=result["injury_risk"],
        injury_probability=result["injury_probability"],
        overtraining=int(result["overtraining"]),
    ))
    return result


def latest_predictions(db: Session) -> dict[int, tuple[models.PredictionLog, date]]:
    """athlete_id -> (prediction of the most recent session, session date)."""
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
    pred, last = latest.get(athlete.id, (None, None))
    data.update(
        latest_readiness=pred.readiness_score if pred else None,
        latest_performance=pred.performance_index if pred else None,
        latest_risk=pred.injury_risk if pred else None,
        last_session=last,
        session_count=counts.get(athlete.id, 0),
    )
    return data


def session_counts(db: Session) -> dict[int, int]:
    return dict(db.execute(
        select(models.TrainingSession.athlete_id, func.count()).group_by(models.TrainingSession.athlete_id)
    ).all())


def trend(athlete: models.Athlete) -> list[dict]:
    """Per-session timeline with session-RPE load and acute:chronic workload ratio.

    sRPE load = duration_min x RPE. Acute = sum of the last 7 days, chronic = weekly
    average of the last 28 days (both inclusive of the session day). ACWR is only
    reported once there are at least 21 days of history, otherwise it is meaningless.
    """
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
            "date": s.session_date,
            "session_id": s.id,
            "srpe_load": round(s.duration_min * s.rpe, 1),
            "acute_7d": round(acute, 1),
            "chronic_28d": round(chronic, 1),
            "acwr": round(acute / chronic, 2) if chronic > 0 and (s.session_date - first).days >= ACWR_MIN_HISTORY_DAYS else None,
            "readiness_score": pred.readiness_score if pred else None,
            "performance_index": pred.performance_index if pred else None,
            "injury_probability": pred.injury_probability if pred else None,
            "injury_risk": pred.injury_risk if pred else None,
            "sprint_100m_s": s.sprint_100m_s,
            "vertical_jump_cm": s.vertical_jump_cm,
            "sleep_hours": s.sleep_hours,
            "wellness": s.wellness,
        })
    return points
