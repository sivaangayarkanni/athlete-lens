"""Deterministic demo roster: eight Tamil Nadu athletes with eight weeks of daily sessions.

Histories come from the same season simulator used for training (with injuries switched off),
steered per athlete so the dashboard shows different situations: a camp load spike, an exam
week with poor sleep, a youth growth spurt, a return from a shoulder injury, etc.
"""
from datetime import date, timedelta

import numpy as np
from sqlalchemy.orm import Session

from . import models
from .injury.simulate import simulate_athlete
from .services import rescore_athlete

DAYS = 56
ATHLETES = [
    dict(a=dict(name="Meenakshi R", sport="Athletics", role="100m / 200m", sex="F", age=19, height_cm=164, weight_kg=52, years_training=5, city="Madurai", academy="SDAT Madurai", previous_injuries=1, injury_history="hamstring", notes="District gold 100m. Hamstring strain last season."),
         plan=dict(sessions_per_week=6, sleep_habit=7.0, capacity=2600, week_overrides={-2: 1.35})),
    dict(a=dict(name="Karthikeyan S", sport="Kabaddi", role="Raider", sex="M", age=21, height_cm=178, weight_kg=72, years_training=6, city="Tiruchengode", academy="Namakkal District Camp", previous_injuries=2, injury_history="knee,ankle", notes="State junior camp this week."),
         plan=dict(sessions_per_week=6, sleep_habit=6.6, capacity=2500, week_overrides={-2: 1.85}, sleep_overrides={-2: -0.6}, matches_from_week=1)),
    dict(a=dict(name="Aishwarya P", sport="Kho-Kho", role="Chaser", sex="F", age=17, height_cm=158, weight_kg=48, years_training=4, city="Karur", academy="Govt. HSS Sports Hostel", previous_injuries=0, growth_cm=1.5, notes="Inter-school captain."),
         plan=dict(sessions_per_week=5, sleep_habit=8.1, capacity=2000, matches_from_week=1)),
    dict(a=dict(name="Mohammed Irfan", sport="Football", role="Winger", sex="M", age=20, height_cm=174, weight_kg=66, years_training=7, city="Coimbatore", academy="SECE Grounds", previous_injuries=2, injury_history="hamstring,groin", nordic_program=1, notes="College first team. On Nordic programme."),
         plan=dict(sessions_per_week=5, sleep_habit=7.4, capacity=2400, matches_from_week=1)),
    dict(a=dict(name="Divya Lakshmi", sport="Hockey", role="Midfield", sex="F", age=18, height_cm=162, weight_kg=54, years_training=5, city="Tirunelveli", academy="SDAT Hockey Academy", previous_injuries=0, notes="Board exams this week."),
         plan=dict(sessions_per_week=5, sleep_habit=7.3, capacity=2200, sleep_overrides={-2: -1.6}, matches_from_week=1)),
    dict(a=dict(name="Vignesh M", sport="Volleyball", role="Spiker", sex="M", age=22, height_cm=186, weight_kg=78, years_training=6, city="Salem", academy="District Stadium", previous_injuries=2, injury_history="knee,shoulder", notes="Patellar tendon history. Trains 7 days a week."),
         plan=dict(sessions_per_week=7, sleep_habit=6.9, capacity=2900, matches_from_week=1)),
    dict(a=dict(name="Harini K", sport="Badminton", role="Singles", sex="F", age=15, height_cm=161, weight_kg=48, years_training=4, city="Coimbatore", academy="Sri Eshwar Indoor", previous_injuries=0, growth_cm=5.0, notes="U-17 circuit. Grew 5 cm in six months."),
         plan=dict(sessions_per_week=6, sleep_habit=7.6, capacity=2300, week_overrides={-2: 1.2})),
    dict(a=dict(name="Surya Prakash", sport="Wrestling", role="74kg", sex="M", age=23, height_cm=172, weight_kg=76, years_training=8, city="Erode", academy="SAI Extension", previous_injuries=3, injury_history="lower_back", notes="Back from a shoulder sprain; cut-weight week."),
         plan=dict(sessions_per_week=6, sleep_habit=6.4, capacity=2700, week_overrides={-2: 1.6}, sleep_overrides={-2: -0.8}),
         injury=dict(region="shoulder", injury_type="ligament_sprain", onset=-38, ret=-24, notes="AC joint sprain in a bout.")),
]


def seed_if_empty(db: Session) -> None:
    if db.query(models.Athlete).count() > 0:
        return
    today = date.today()
    start = today - timedelta(days=DAYS - 1)
    for i, spec in enumerate(ATHLETES):
        rng = np.random.default_rng(1000 + i)
        a = models.Athlete(**spec["a"])
        db.add(a)
        db.flush()
        profile = {"sport": a.sport, "sex": a.sex, "age": a.age, "height_cm": a.height_cm, "weight_kg": a.weight_kg,
                   "years_training": a.years_training, "growth_cm": a.growth_cm or 0.0,
                   "nordic": a.nordic_program, "adductor": a.adductor_program}
        sim = simulate_athlete(rng, profile, season_days=DAYS - 28, allow_injuries=False, plan=spec["plan"])
        d = sim["daily"]
        blocked = set()
        if "injury" in spec:
            inj = spec["injury"]
            onset, ret = today + timedelta(days=inj["onset"]), today + timedelta(days=inj["ret"])
            db.add(models.InjuryRecord(athlete_id=a.id, region=inj["region"], injury_type=inj["injury_type"],
                                       onset_date=onset, return_date=ret, notes=inj["notes"]))
            blocked = {(onset - start).days + k for k in range((ret - onset).days)}
        sess_flags = [bool(d["session"][k]) and k not in blocked for k in range(DAYS)]
        for k in range(DAYS):
            if not sess_flags[k]:
                continue
            last7 = sum(sess_flags[max(0, k - 6):k + 1])
            nz = lambda v: None if np.isnan(v) else float(v)  # noqa: E731
            db.add(models.TrainingSession(
                athlete_id=a.id, session_date=start + timedelta(days=k),
                duration_min=float(d["duration"][k]), distance_km=float(d["distance"][k]),
                sprint_100m_s=nz(d["sprint"][k]), vertical_jump_cm=nz(d["jump"][k]),
                resting_hr=float(d["rhr"][k]) if not np.isnan(d["rhr"][k]) else 62.0,
                session_hr_avg=float(np.clip(150 + 3 * (d["rpe"][k] - 6) + rng.normal(0, 4), 100, 200)),
                rpe=float(d["rpe"][k]), sleep_hours=float(d["sleep"][k]), wellness=float(d["wellness"][k]),
                sessions_last_7=int(last7), rest_days_last_7=int(7 - last7),
                session_type="match" if d["match"][k] else "training", asymmetry_pct=nz(d["asymmetry"][k]),
                notes="Match" if d["match"][k] else "",
            ))
        db.flush()
        db.refresh(a)
        rescore_athlete(db, a)
    db.commit()
