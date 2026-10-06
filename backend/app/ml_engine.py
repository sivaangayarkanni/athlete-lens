"""Performance / readiness engine (v1, synthetic) + glue to the per-region injury engine.

* ``GradientBoostingRegressor`` -> performance index (0-100), trained on a synthetic cohort.
* Injury risk now comes from ``backend.app.injury`` (per-region, calibrated, history-aware).
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import settings
from .injury.catalog import PREVENTION, REGION_LABELS, SPORTS

log = logging.getLogger("athlete_lens.ml")

BUNDLED_DIR = Path(__file__).resolve().parent / "artifacts"
COHORT_SIZE = 2400
TEST_FRACTION = 0.2
SEED = 42
DEFAULT_SPRINT = {"F": 14.4, "M": 13.1}
DEFAULT_JUMP = {"F": 38.0, "M": 46.0}

FEATURE_NUM = [
    "age", "years_training", "height_cm", "weight_kg", "bmi", "previous_injuries",
    "duration_min", "distance_km", "sprint_100m_s", "vertical_jump_cm", "resting_hr",
    "session_hr_avg", "hr_reserve_proxy", "rpe", "sleep_hours", "wellness",
    "sessions_last_7", "rest_days_last_7", "acute_load", "is_female",
]
FEATURE_NAMES = FEATURE_NUM + [f"sport={s}" for s in SPORTS]
INPUT_FIELDS = [
    "age", "sex", "sport", "years_training", "height_cm", "weight_kg", "previous_injuries",
    "duration_min", "distance_km", "sprint_100m_s", "vertical_jump_cm", "resting_hr",
    "session_hr_avg", "rpe", "sleep_hours", "wellness", "sessions_last_7", "rest_days_last_7",
]


def enrich(row: dict) -> dict:
    out = dict(row)
    sex = "F" if str(out.get("sex", "F")).upper().startswith("F") else "M"
    if out.get("sprint_100m_s") is None:
        out["sprint_100m_s"] = DEFAULT_SPRINT[sex]
    if out.get("vertical_jump_cm") is None:
        out["vertical_jump_cm"] = DEFAULT_JUMP[sex]
    out["bmi"] = float(out["weight_kg"]) / ((float(out["height_cm"]) / 100) ** 2)
    out["hr_reserve_proxy"] = float(out["session_hr_avg"]) - float(out["resting_hr"])
    out["acute_load"] = float(out["duration_min"]) * float(out["rpe"]) * max(1, int(out["sessions_last_7"])) / 7.0
    out["is_female"] = 1.0 if sex == "F" else 0.0
    if out.get("sport") not in SPORTS:
        out["sport"] = "Athletics"
    return out


def vectorize(row: dict) -> list[float]:
    return [float(row[k]) for k in FEATURE_NUM] + [1.0 if row["sport"] == s else 0.0 for s in SPORTS]


def _cohort(n: int = COHORT_SIZE, seed: int = SEED):
    """Synthetic cohort for the performance index. Returns (X, y_perf)."""
    rng = np.random.default_rng(seed)
    X, y_perf = [], []
    for _ in range(n):
        sex = rng.choice(["F", "M"])
        sport = rng.choice(SPORTS)
        age = int(rng.integers(13, 28))
        years = float(max(0.3, rng.normal(age - 14, 1.8)))
        height = float(np.clip(rng.normal(160 if sex == "F" else 172, 8), 145, 198))
        weight = float(np.clip(rng.normal(52 if sex == "F" else 64, 8), 38, 105))
        sprint = float(np.clip(rng.normal(14.4 if sex == "F" else 13.1, 1.1), 10.6, 18.5))
        jump = float(np.clip(rng.normal(38 if sex == "F" else 46, 8), 22, 78))
        rhr = float(np.clip(rng.normal(66 if sex == "F" else 62, 7), 44, 92))
        sessions7 = int(np.clip(rng.integers(2, 9), 1, 12))
        rest7 = int(np.clip(7 - sessions7 + int(rng.integers(-1, 2)), 0, 6))
        duration = float(np.clip(rng.normal(70, 18), 25, 160))
        distance = float(np.clip(rng.normal(5.2, 2.4), 0.2, 22))
        rpe = float(np.clip(rng.normal(6.4, 1.5), 2, 10))
        sleep = float(np.clip(rng.normal(6.8, 1.1), 3.5, 10.5))
        wellness = float(np.clip(rng.normal(6.9, 1.4), 2, 10))
        inj = int(np.clip(rng.poisson(0.6), 0, 6))
        shr = float(np.clip(rhr + rng.normal(78, 14), 95, 195))
        row = enrich(dict(
            age=age, sex=sex, sport=sport, years_training=years, height_cm=height, weight_kg=weight,
            previous_injuries=inj, duration_min=duration, distance_km=distance, sprint_100m_s=sprint,
            vertical_jump_cm=jump, resting_hr=rhr, session_hr_avg=shr, rpe=rpe, sleep_hours=sleep,
            wellness=wellness, sessions_last_7=sessions7, rest_days_last_7=rest7,
        ))
        speed = np.clip((18 - sprint) / 7.5, 0, 1)
        power = np.clip((jump - 20) / 50, 0, 1)
        fatigue = np.clip((row["acute_load"] - 180) / 280, 0, 1)
        sleep_pen = np.clip((7.2 - sleep) / 4, 0, 1)
        perf = 100 * np.clip(
            0.28 * speed + 0.2 * power + 0.16 * np.clip((75 - rhr) / 40, 0, 1) + 0.12 * (wellness / 10)
            + 0.1 * np.clip(years / 8, 0, 1) - 0.18 * fatigue - 0.1 * sleep_pen + rng.normal(0, 0.04),
            0.12, 0.97,
        )
        X.append(vectorize(row))
        y_perf.append(float(round(perf, 2)))
    return np.asarray(X, dtype=float), np.asarray(y_perf, dtype=float)


class AthleteLensModel:
    REG = "performance_gbr.joblib"
    META = "meta.joblib"

    def __init__(self, model_dir: str | Path | None = None):
        self.dir = Path(model_dir or settings.model_dir)
        self.reg: Pipeline | None = None
        self.meta: dict = {}
        self.source: str = "unloaded"
        self._lock = threading.Lock()

    def train(self, n: int = COHORT_SIZE, out_dir: str | Path | None = None) -> dict:
        t0 = time.perf_counter()
        X, y = _cohort(n)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_FRACTION, random_state=SEED)
        reg = Pipeline([("scale", StandardScaler()), ("model", GradientBoostingRegressor(
            n_estimators=120, max_depth=3, learning_rate=0.08, random_state=SEED))]).fit(X_tr, y_tr)
        pred = reg.predict(X_te)
        self.reg = reg
        self.meta = {
            "cohort_size": int(n), "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)), "seed": SEED,
            "sklearn_version": sklearn.__version__, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "train_seconds": round(time.perf_counter() - t0, 2),
            "metrics": {"performance": {
                "holdout_mae": round(float(mean_absolute_error(y_te, pred)), 2),
                "holdout_r2": round(float(r2_score(y_te, pred)), 3),
                "train_mae": round(float(mean_absolute_error(y_tr, reg.predict(X_tr))), 2),
                "baseline_mean_mae": round(float(np.mean(np.abs(y_te - y_tr.mean()))), 2),
            }},
            "features": FEATURE_NAMES,
            "performance_model": "GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.08)",
        }
        target = Path(out_dir) if out_dir else self.dir
        try:
            target.mkdir(parents=True, exist_ok=True)
            joblib.dump(reg, target / self.REG, compress=3)
            joblib.dump(self.meta, target / self.META)
        except OSError as exc:
            log.warning("could not persist models to %s: %s", target, exc)
        self.source = f"trained:{target}"
        return self.meta

    def _try_load(self, directory: Path) -> bool:
        paths = [directory / self.REG, directory / self.META]
        if not all(p.exists() for p in paths):
            return False
        try:
            meta = joblib.load(paths[1])
            if meta.get("sklearn_version") != sklearn.__version__ or meta.get("features") != FEATURE_NAMES:
                return False
            self.reg, self.meta = joblib.load(paths[0]), meta
            self.source = str(directory)
            return True
        except Exception as exc:
            log.warning("failed loading artifacts from %s: %s", directory, exc)
            return False

    def load(self) -> None:
        if self.ready:
            return
        with self._lock:
            if self.ready:
                return
            if self._try_load(self.dir) or self._try_load(BUNDLED_DIR):
                return
            self.train()

    @property
    def ready(self) -> bool:
        return self.reg is not None

    # ------------------------------------------------------------------ inference
    def analyze(self, payload: dict, injury: dict | None = None) -> dict:
        """Readiness / performance analysis. `injury` is the per-region summary from the injury engine."""
        self.load()
        row = enrich({k: payload.get(k) for k in INPUT_FIELDS if k in payload})
        X = np.asarray([vectorize(row)], dtype=float)
        perf = float(np.clip(self.reg.predict(X)[0], 8, 98))
        load = float(np.clip(row["acute_load"] / 3.2, 0, 100))
        recovery = float(np.clip(
            12 * row["sleep_hours"] + 6 * row["wellness"] + 8 * row["rest_days_last_7"]
            - 4 * max(0, row["rpe"] - 6) - 0.08 * row["acute_load"], 8, 98))
        speed = float(np.clip((18.2 - row["sprint_100m_s"]) / 7.4 * 100, 5, 99))
        power = float(np.clip((row["vertical_jump_cm"] - 18) / 52 * 100, 5, 99))
        readiness = float(np.clip(0.42 * recovery + 0.22 * (100 - min(load, 100)) + 0.18 * perf + 0.18 * row["wellness"] * 10, 6, 98))
        if injury:
            p_inj, risk = injury["overall"]["probability"], injury["overall"]["band"]
        else:  # rule-based fallback (only used if the injury engine is unavailable)
            p_inj = float(np.clip(0.03 + 0.0002 * max(0, row["acute_load"] - 250) + 0.01 * row["previous_injuries"], 0, 0.5))
            risk = "high" if row["rest_days_last_7"] <= 1 and row["rpe"] >= 8 else "low"
        overtraining = bool(row["sessions_last_7"] >= 7 and row["sleep_hours"] < 6.5 and row["rpe"] >= 7.5)
        return {
            "performance_index": round(perf, 1),
            "readiness_score": round(readiness, 1),
            "readiness_band": "green" if readiness >= 65 else "amber" if readiness >= 45 else "red",
            "injury_risk": risk,
            "injury_probability": round(p_inj, 4),
            "overtraining": overtraining,
            "load_score": round(min(load, 100), 1),
            "recovery_score": round(recovery, 1),
            "speed_score": round(speed, 1),
            "power_score": round(power, 1),
            "explanations": self._explain(row, p_inj, load, recovery, injury),
            "recommendations": self._recs(row, risk, overtraining, injury),
            "plan_72h": self._plan(row, risk, readiness, overtraining, injury),
            "feature_importance": self.importance(injury),
            "injury": injury,
        }

    @staticmethod
    def importance(injury: dict | None = None) -> list[dict]:
        if injury and injury.get("drivers") and "any" in injury["drivers"]:
            return [{"feature": d["label"], "weight": d["contribution"]} for d in injury["drivers"]["any"]["drivers"][:8]]
        return []

    @staticmethod
    def _explain(row, p_inj, load, recovery, injury) -> list[str]:
        notes = []
        if row["sleep_hours"] < 7:
            notes.append("Sleep is below 7 h; athletes sleeping <8 h had 1.7x the injury likelihood in one adolescent cohort (Milewski 2014).")
        if row["rest_days_last_7"] <= 1:
            notes.append("Almost no rest day in the last week.")
        if row["rpe"] >= 8:
            notes.append("Session felt very hard (RPE >= 8).")
        if injury and injury.get("load", {}).get("acwr"):
            a = injury["load"]["acwr"]
            if a >= 1.5:
                notes.append(f"Acute:chronic workload ratio is {a:.2f}, a load spike (>1.5).")
            elif a < 0.8:
                notes.append(f"Acute:chronic workload ratio is {a:.2f}: under-loaded relative to the usual level.")
        if injury:
            top = [r for r in injury["regions"] if r["region"] == injury["top_regions"][0]][0]
            notes.append(f"Highest relative region risk: {top['label']} at {top['probability']:.1%} over 7 days "
                         f"({top['relative_risk']:.1f}x the sport average), most likely {top.get('likely_type_label', '').lower()}.")
        if not notes:
            notes.append("Load, sleep and wellness are balanced.")
        notes.append(f"Overall 7-day injury probability {p_inj:.1%}; load {load:.0f}/100, recovery {recovery:.0f}/100.")
        return notes[:6]

    @staticmethod
    def _recs(row, risk, overtraining, injury) -> list[dict]:
        recs = []
        if risk == "high" or overtraining:
            recs.append({"title": "Deload this block", "detail": "Drop high-intensity reps. Keep mobility + easy aerobic 25-35 min.", "priority": "now", "tag": "injury"})
        elif risk == "moderate":
            recs.append({"title": "Cap intensity", "detail": "One quality speed or power session only.", "priority": "this_week", "tag": "load"})
        else:
            recs.append({"title": "Green light for quality work", "detail": "Keep the planned speed day. Protect sleep at 8h after it.", "priority": "this_week", "tag": "performance"})
        if injury:
            top = injury["top_regions"][0]
            recs.append({"title": f"Protect the {REGION_LABELS[top].split(' (')[0].lower()}",
                         "detail": PREVENTION[top]["prevention"][0], "priority": "this_week", "tag": top})
        if row["sleep_hours"] < 7.5:
            recs.append({"title": "Fix the night before the session", "detail": "Target 8 hours. A 20-min walk after dinner helps more than extra drills.", "priority": "now", "tag": "recovery"})
        if row["rest_days_last_7"] <= 1:
            recs.append({"title": "Book a full rest day", "detail": "At least one zero-training day in the next 72 hours.", "priority": "now", "tag": "recovery"})
        recs.append({"title": "Weekly test you can do on any ground", "detail": "Record 30m fly, vertical jump, single-leg hop asymmetry and wellness (1-10).", "priority": "monitor", "tag": "test"})
        return recs[:5]

    @staticmethod
    def _plan(row, risk, readiness, overtraining, injury) -> list[dict]:
        focus_region = REGION_LABELS[injury["top_regions"][0]].split(" (")[0].lower() if injury else None
        prev = PREVENTION[injury["top_regions"][0]]["prevention"][0] if injury else ""
        if risk == "high" or overtraining:
            days = [
                ("Rest / recovery", "rest", 0, "Full rest or 20 min walk + mobility. Sleep 8h."),
                ("Easy aerobic + prevention", "low", 30, f"Zone-1 jog or cycle, RPE 3-4. {prev}." if prev else "Zone-1 aerobic, RPE 3-4."),
                ("Technique only", "low", 40, "Drills at 60% effort, no max sprints or jumps. Re-check wellness."),
            ]
        elif risk == "moderate" or readiness < 55:
            days = [
                ("Moderate session", "moderate", 50, "Tempo work at RPE 5-6. Cut planned volume by ~20%."),
                ("Recovery + prevention", "low", 30, f"Mobility and light aerobic. {prev}." if prev else "Mobility, light aerobic."),
                ("One quality block", "moderate", 55, f"Short speed or power block (≤6 reps) if wellness ≥ 6; monitor the {focus_region}." if focus_region else "Short quality block."),
            ]
        else:
            days = [
                ("Quality speed / power", "high", 70, "Planned speed day: full recoveries between reps."),
                ("Recovery", "low", 35, "Easy aerobic + mobility. Protect 8h sleep."),
                ("Skill + strength", "moderate", 60, f"Sport skill work and strength circuit incl. {prev.split(':')[0].lower()}." if prev else "Skill + strength."),
            ]
        return [{"day": i + 1, "focus": f, "intensity": it, "minutes": m, "detail": d} for i, (f, it, m, d) in enumerate(days)]


engine_singleton = AthleteLensModel()
