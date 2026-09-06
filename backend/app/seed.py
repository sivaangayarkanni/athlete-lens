from datetime import date, timedelta

from sqlalchemy.orm import Session

from . import models
from .ml_engine import engine_singleton


ATHLETES = [
    dict(name="Meenakshi R", sport="Athletics", role="100m / 200m", sex="F", age=19, height_cm=164, weight_kg=52, years_training=5, city="Madurai", academy="SDAT Madurai", previous_injuries=1, notes="District gold 100m."),
    dict(name="Karthikeyan S", sport="Kabaddi", role="Raider", sex="M", age=21, height_cm=178, weight_kg=72, years_training=6, city="Tiruchengode", academy="Namakkal District Camp", previous_injuries=2, notes="State junior camp."),
    dict(name="Aishwarya P", sport="Kho-Kho", role="Chaser", sex="F", age=17, height_cm=158, weight_kg=48, years_training=4, city="Karur", academy="Govt. HSS Sports Hostel", previous_injuries=0, notes="Inter-school captain."),
    dict(name="Mohammed Irfan", sport="Football", role="Winger", sex="M", age=20, height_cm=174, weight_kg=66, years_training=7, city="Coimbatore", academy="SECE Grounds", previous_injuries=1, notes="College first team."),
    dict(name="Divya Lakshmi", sport="Hockey", role="Midfield", sex="F", age=18, height_cm=162, weight_kg=54, years_training=5, city="Tirunelveli", academy="SDAT Hockey Academy", previous_injuries=0, notes="Needs speed endurance."),
    dict(name="Vignesh M", sport="Volleyball", role="Spiker", sex="M", age=22, height_cm=186, weight_kg=78, years_training=6, city="Salem", academy="District Stadium", previous_injuries=1, notes="Jump load management."),
    dict(name="Harini K", sport="Badminton", role="Singles", sex="F", age=16, height_cm=161, weight_kg=50, years_training=4, city="Coimbatore", academy="Sri Eshwar Indoor", previous_injuries=0, notes="U-17 circuit."),
    dict(name="Surya Prakash", sport="Wrestling", role="74kg", sex="M", age=23, height_cm=172, weight_kg=76, years_training=8, city="Erode", academy="SAI Extension", previous_injuries=3, notes="Cut-weight weeks are risky."),
]


def seed_if_empty(db: Session) -> None:
    engine_singleton.load()
    if db.query(models.Athlete).count() > 0:
        return
    today = date.today()
    for i, raw in enumerate(ATHLETES):
        a = models.Athlete(**raw)
        db.add(a)
        db.flush()
        for d in range(10, -1, -2):
            tired = d in (2, 0)
            sess = models.TrainingSession(
                athlete_id=a.id,
                session_date=today - timedelta(days=d),
                duration_min=80 if tired else 62,
                distance_km=7.5 if a.sport in {"Athletics", "Football", "Hockey"} else 2.4,
                sprint_100m_s=13.1 + (0.6 if a.sex == "F" else 0) + (0.5 if tired else 0),
                vertical_jump_cm=(40 if a.sex == "F" else 48) - (4 if tired else 0) + i,
                resting_hr=62 + (8 if tired else 0),
                session_hr_avg=150 + (12 if tired else 0),
                rpe=8.2 if tired else 6.1,
                sleep_hours=5.8 if tired else 7.4,
                wellness=5 if tired else 8,
                sessions_last_7=7 if tired else 4,
                rest_days_last_7=0 if tired else 2,
                notes="Camp block" if tired else "Quality session",
            )
            db.add(sess)
            db.flush()
            payload = {
                "age": a.age,
                "sex": a.sex,
                "sport": a.sport,
                "years_training": a.years_training,
                "height_cm": a.height_cm,
                "weight_kg": a.weight_kg,
                "previous_injuries": a.previous_injuries,
                "duration_min": sess.duration_min,
                "distance_km": sess.distance_km,
                "sprint_100m_s": sess.sprint_100m_s,
                "vertical_jump_cm": sess.vertical_jump_cm,
                "resting_hr": sess.resting_hr,
                "session_hr_avg": sess.session_hr_avg,
                "rpe": sess.rpe,
                "sleep_hours": sess.sleep_hours,
                "wellness": sess.wellness,
                "sessions_last_7": sess.sessions_last_7,
                "rest_days_last_7": sess.rest_days_last_7,
            }
            pred = engine_singleton.analyze(payload)
            db.add(
                models.PredictionLog(
                    athlete_id=a.id,
                    session_id=sess.id,
                    performance_index=pred["performance_index"],
                    readiness_score=pred["readiness_score"],
                    injury_risk=pred["injury_risk"],
                    injury_probability=pred["injury_probability"],
                    overtraining=int(pred["overtraining"]),
                )
            )
    db.commit()
