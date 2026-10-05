"""Deterministic demo roster: eight Tamil Nadu athletes with four weeks of sessions."""
from datetime import date, timedelta

import numpy as np
from sqlalchemy.orm import Session

from . import models
from .services import score_session

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

# Training "story" for the last week per athlete: how hard the final block is (0 = fresh, 1 = camp overload).
BLOCK_STRESS = [0.2, 0.9, 0.1, 0.5, 0.3, 0.6, 0.0, 1.0]
ENDURANCE_SPORTS = {"Athletics", "Football", "Hockey", "Kho-Kho"}


def seed_if_empty(db: Session) -> None:
    if db.query(models.Athlete).count() > 0:
        return
    today = date.today()
    for i, raw in enumerate(ATHLETES):
        rng = np.random.default_rng(100 + i)
        a = models.Athlete(**raw)
        db.add(a)
        db.flush()
        stress = BLOCK_STRESS[i]
        female = a.sex == "F"
        # 14 sessions across 27 days (every other day), last week ramps with `stress`.
        for d in range(26, -1, -2):
            late = d <= 6
            s = stress if late else 0.15
            sess = models.TrainingSession(
                athlete_id=a.id,
                session_date=today - timedelta(days=d),
                duration_min=round(float(60 + 35 * s + rng.normal(0, 5)), 0),
                distance_km=round(float((6.5 if a.sport in ENDURANCE_SPORTS else 2.5) + rng.normal(0, 0.6)), 1),
                sprint_100m_s=round(float((13.7 if female else 12.6) + 0.5 * s + rng.normal(0, 0.12)), 2),
                vertical_jump_cm=round(float((40 if female else 50) + i * 0.6 - 4 * s + rng.normal(0, 1.2)), 1),
                resting_hr=round(float(60 + 9 * s + rng.normal(0, 1.5)), 0),
                session_hr_avg=round(float(146 + 14 * s + rng.normal(0, 3)), 0),
                rpe=round(float(min(10, 5.8 + 2.6 * s + rng.normal(0, 0.3))), 1),
                sleep_hours=round(float(7.6 - 1.9 * s + rng.normal(0, 0.25)), 1),
                wellness=round(float(min(10, 8 - 3.2 * s + rng.normal(0, 0.4))), 1),
                sessions_last_7=int(round(4 + 3 * s)),
                rest_days_last_7=int(round(2 - 2 * s)),
                notes="Camp block" if late and s >= 0.6 else "Quality session",
            )
            db.add(sess)
            db.flush()
            score_session(db, a, sess)
    db.commit()
