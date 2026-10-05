"""Athlete Lens ML engine.

Two scikit-learn models trained on a physiologically-plausible synthetic cohort:

* ``RandomForestClassifier``  -> injury-risk probability
* ``GradientBoostingRegressor`` -> performance index (0-100)

Features are built as a plain numeric matrix (sport is one-hot encoded by hand),
so the runtime only needs numpy + scikit-learn (no pandas) — this keeps the
serverless bundle small.

Artifacts are resolved in this order:
1. ``settings.model_dir`` (writable, e.g. /tmp) if it already holds artifacts
2. the bundled, pre-trained artifacts in ``backend/app/artifacts``
3. train from scratch and write to ``settings.model_dir``

Artifacts trained with a different scikit-learn version are ignored and retrained.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import settings

log = logging.getLogger("athlete_lens.ml")

BUNDLED_DIR = Path(__file__).resolve().parent / "artifacts"
COHORT_SIZE = 2400
TEST_FRACTION = 0.2
SEED = 42

FEATURE_NUM = [
    "age", "years_training", "height_cm", "weight_kg", "bmi", "previous_injuries",
    "duration_min", "distance_km", "sprint_100m_s", "vertical_jump_cm", "resting_hr",
    "session_hr_avg", "hr_reserve_proxy", "rpe", "sleep_hours", "wellness",
    "sessions_last_7", "rest_days_last_7", "acute_load", "is_female",
]
SPORTS = ["Athletics", "Kabaddi", "Kho-Kho", "Football", "Hockey", "Volleyball", "Badminton", "Wrestling"]
FEATURE_NAMES = FEATURE_NUM + [f"sport={s}" for s in SPORTS]

INPUT_FIELDS = [
    "age", "sex", "sport", "years_training", "height_cm", "weight_kg", "previous_injuries",
    "duration_min", "distance_km", "sprint_100m_s", "vertical_jump_cm", "resting_hr",
    "session_hr_avg", "rpe", "sleep_hours", "wellness", "sessions_last_7", "rest_days_last_7",
]


def enrich(row: dict) -> dict:
    out = dict(row)
    out["bmi"] = float(out["weight_kg"]) / ((float(out["height_cm"]) / 100) ** 2)
    out["hr_reserve_proxy"] = float(out["session_hr_avg"]) - float(out["resting_hr"])
    out["acute_load"] = float(out["duration_min"]) * float(out["rpe"]) * max(1, int(out["sessions_last_7"])) / 7.0
    out["is_female"] = 1.0 if str(out.get("sex", "F")).upper().startswith("F") else 0.0
    if out.get("sport") not in SPORTS:
        out["sport"] = "Athletics"
    return out


def vectorize(row: dict) -> list[float]:
    """Enriched row -> numeric feature vector (numeric features + sport one-hot)."""
    return [float(row[k]) for k in FEATURE_NUM] + [1.0 if row["sport"] == s else 0.0 for s in SPORTS]


def _cohort(n: int = COHORT_SIZE, seed: int = SEED):
    """Synthetic but physiologically-plausible cohort. Returns (X, y_injury, y_perf)."""
    rng = np.random.default_rng(seed)
    X, y_inj, y_perf = [], [], []
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
        logit = (-2.2 + 0.55 * inj + 1.1 * fatigue + 0.7 * sleep_pen + 0.35 * (rpe >= 8)
                 + 0.45 * (rest7 <= 1) + 0.25 * (sessions7 >= 7) + rng.normal(0, 0.35))
        p = 1 / (1 + np.exp(-logit))
        X.append(vectorize(row))
        y_perf.append(float(round(perf, 2)))
        y_inj.append(int(rng.random() < p))
    return np.asarray(X, dtype=float), np.asarray(y_inj, dtype=int), np.asarray(y_perf, dtype=float)


def _new_clf() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", RandomForestClassifier(
        n_estimators=140, max_depth=9, min_samples_leaf=8, class_weight="balanced", random_state=SEED, n_jobs=1))])


def _new_reg() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", GradientBoostingRegressor(
        n_estimators=120, max_depth=3, learning_rate=0.08, random_state=SEED))])


class AthleteLensModel:
    CLF = "injury_rf.joblib"
    REG = "performance_gbr.joblib"
    META = "meta.joblib"

    def __init__(self, model_dir: str | Path | None = None):
        self.dir = Path(model_dir or settings.model_dir)
        self.clf: Pipeline | None = None
        self.reg: Pipeline | None = None
        self.meta: dict = {}
        self.source: str = "unloaded"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ training
    def train(self, n: int = COHORT_SIZE, out_dir: str | Path | None = None) -> dict:
        t0 = time.perf_counter()
        X, y_inj, y_perf = _cohort(n)
        X_tr, X_te, yi_tr, yi_te, yp_tr, yp_te = train_test_split(
            X, y_inj, y_perf, test_size=TEST_FRACTION, random_state=SEED, stratify=y_inj)
        clf, reg = _new_clf(), _new_reg()
        clf.fit(X_tr, yi_tr)
        reg.fit(X_tr, yp_tr)

        proba = clf.predict_proba(X_te)[:, 1]
        pred = clf.predict(X_te)
        perf_pred = reg.predict(X_te)
        majority = int(np.bincount(yi_tr).argmax())
        metrics = {
            "injury": {
                "holdout_accuracy": round(float(accuracy_score(yi_te, pred)), 3),
                "holdout_balanced_accuracy": round(float(balanced_accuracy_score(yi_te, pred)), 3),
                "holdout_recall": round(float(recall_score(yi_te, pred)), 3),
                "holdout_roc_auc": round(float(roc_auc_score(yi_te, proba)), 3),
                "holdout_brier": round(float(brier_score_loss(yi_te, proba)), 3),
                "train_accuracy": round(float(accuracy_score(yi_tr, clf.predict(X_tr))), 3),
                "baseline_majority_accuracy": round(float((yi_te == majority).mean()), 3),
                "positive_rate": round(float(y_inj.mean()), 3),
            },
            "performance": {
                "holdout_mae": round(float(mean_absolute_error(yp_te, perf_pred)), 2),
                "holdout_r2": round(float(r2_score(yp_te, perf_pred)), 3),
                "train_mae": round(float(mean_absolute_error(yp_tr, reg.predict(X_tr))), 2),
                "baseline_mean_mae": round(float(np.mean(np.abs(yp_te - yp_tr.mean()))), 2),
            },
        }
        self.clf, self.reg = clf, reg
        self.meta = {
            "cohort_size": int(n),
            "train_rows": int(len(X_tr)),
            "test_rows": int(len(X_te)),
            "seed": SEED,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "train_seconds": round(time.perf_counter() - t0, 2),
            "metrics": metrics,
            "sports": SPORTS,
            "features": FEATURE_NAMES,
            "injury_model": "RandomForestClassifier(n_estimators=140, max_depth=9, min_samples_leaf=8, class_weight=balanced)",
            "performance_model": "GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.08)",
        }
        target = Path(out_dir) if out_dir else self.dir
        try:
            target.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, target / self.CLF, compress=3)
            joblib.dump(reg, target / self.REG, compress=3)
            joblib.dump(self.meta, target / self.META)
        except OSError as exc:  # read-only FS: keep the in-memory model
            log.warning("could not persist models to %s: %s", target, exc)
        self.source = f"trained:{target}"
        return self.meta

    # ------------------------------------------------------------------ loading
    def _try_load(self, directory: Path) -> bool:
        paths = [directory / self.CLF, directory / self.REG, directory / self.META]
        if not all(p.exists() for p in paths):
            return False
        try:
            meta = joblib.load(paths[2])
            if meta.get("sklearn_version") != sklearn.__version__ or meta.get("features") != FEATURE_NAMES:
                log.info("artifacts in %s are stale (sklearn %s); retraining", directory, meta.get("sklearn_version"))
                return False
            self.clf, self.reg, self.meta = joblib.load(paths[0]), joblib.load(paths[1]), meta
            self.source = str(directory)
            return True
        except Exception as exc:  # corrupt / incompatible pickle
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
        return self.clf is not None and self.reg is not None

    # ------------------------------------------------------------------ inference
    def analyze(self, payload: dict) -> dict:
        self.load()
        row = enrich({k: payload[k] for k in INPUT_FIELDS if k in payload})
        X = np.asarray([vectorize(row)], dtype=float)
        p_inj = float(self.clf.predict_proba(X)[0, 1])
        perf = float(np.clip(self.reg.predict(X)[0], 8, 98))
        load = float(np.clip(row["acute_load"] / 3.2, 0, 100))
        recovery = float(np.clip(
            12 * row["sleep_hours"] + 6 * row["wellness"] + 8 * row["rest_days_last_7"]
            - 4 * max(0, row["rpe"] - 6) - 0.08 * row["acute_load"], 8, 98))
        speed = float(np.clip((18.2 - row["sprint_100m_s"]) / 7.4 * 100, 5, 99))
        power = float(np.clip((row["vertical_jump_cm"] - 18) / 52 * 100, 5, 99))
        readiness = float(np.clip(0.42 * recovery + 0.22 * (100 - min(load, 100)) + 0.18 * perf + 0.18 * row["wellness"] * 10, 6, 98))
        if p_inj >= 0.55 or (row["rest_days_last_7"] <= 1 and row["rpe"] >= 8):
            risk = "high"
        elif p_inj >= 0.32:
            risk = "moderate"
        else:
            risk = "low"
        overtraining = bool(row["sessions_last_7"] >= 7 and row["sleep_hours"] < 6.5 and row["rpe"] >= 7.5)
        return {
            "performance_index": round(perf, 1),
            "readiness_score": round(readiness, 1),
            "readiness_band": "green" if readiness >= 65 else "amber" if readiness >= 45 else "red",
            "injury_risk": risk,
            "injury_probability": round(p_inj, 3),
            "overtraining": overtraining,
            "load_score": round(min(load, 100), 1),
            "recovery_score": round(recovery, 1),
            "speed_score": round(speed, 1),
            "power_score": round(power, 1),
            "explanations": self._explain(row, p_inj, load, recovery),
            "recommendations": self._recs(row, risk, overtraining),
            "plan_72h": self._plan(row, risk, readiness, overtraining),
            "feature_importance": self.importance(8),
        }

    def importance(self, top: int = 8) -> list[dict]:
        if self.clf is None:
            return []
        weights = self.clf.named_steps["model"].feature_importances_
        pairs = sorted(zip(FEATURE_NAMES, weights), key=lambda x: -x[1])[:top]
        return [{"feature": n.replace("_", " "), "weight": round(float(v), 3)} for n, v in pairs]

    @staticmethod
    def _explain(row, p_inj, load, recovery) -> list[str]:
        notes = []
        if row["sleep_hours"] < 6.5:
            notes.append("Sleep is below the 7-hour recovery line.")
        if row["rest_days_last_7"] <= 1:
            notes.append("Almost no rest day in the last week.")
        if row["rpe"] >= 8:
            notes.append("Session felt very hard (RPE >= 8).")
        if row["previous_injuries"] >= 2:
            notes.append("Injury history is non-zero — progress load slowly.")
        if row["acute_load"] > 280:
            notes.append("Acute training load is spiked. Cut volume 20-30% for 4-5 days.")
        if not notes:
            notes.append("Load, sleep and wellness are balanced.")
        notes.append(f"Model injury probability is {p_inj:.0%} with load {load:.0f}/100 and recovery {recovery:.0f}/100.")
        return notes[:5]

    @staticmethod
    def _recs(row, risk, overtraining) -> list[dict]:
        recs = []
        if risk == "high" or overtraining:
            recs.append({"title": "Deload this block", "detail": "Drop high-intensity reps. Keep mobility + easy aerobic 25-35 min.", "priority": "now", "tag": "injury"})
        elif risk == "moderate":
            recs.append({"title": "Cap intensity", "detail": "One quality speed or power session only.", "priority": "this_week", "tag": "load"})
        else:
            recs.append({"title": "Green light for quality work", "detail": "Keep the planned speed day. Protect sleep at 7.5h+ after it.", "priority": "this_week", "tag": "performance"})
        if row["sleep_hours"] < 7:
            recs.append({"title": "Fix the night before the session", "detail": "Target 7.5 hours. A 20-min walk after dinner helps more than extra drills.", "priority": "now", "tag": "recovery"})
        if row["rest_days_last_7"] <= 1:
            recs.append({"title": "Book a full rest day", "detail": "At least one zero-training day in the next 72 hours.", "priority": "now", "tag": "recovery"})
        recs.append({"title": "Weekly test you can do on any ground", "detail": "Record 30m fly, standing long jump, and wellness (1-10).", "priority": "monitor", "tag": "test"})
        return recs[:4]

    @staticmethod
    def _plan(row, risk, readiness, overtraining) -> list[dict]:
        """Rule-based 72-hour coaching plan derived from the model outputs."""
        if risk == "high" or overtraining:
            days = [
                ("Rest / recovery", "rest", 0, "Full rest or 20 min walk + mobility. Sleep 8h."),
                ("Easy aerobic", "low", 30, "Zone-1 jog or cycle, RPE 3-4. Core + hip stability."),
                ("Technique only", "low", 40, "Drills at 60% effort, no max sprints or jumps. Re-check wellness."),
            ]
        elif risk == "moderate" or readiness < 55:
            days = [
                ("Moderate session", "moderate", 50, "Tempo work at RPE 5-6. Cut planned volume by ~20%."),
                ("Recovery", "low", 30, "Mobility, light aerobic, stretching. Log sleep."),
                ("One quality block", "moderate", 55, "Short speed or power block (≤6 reps) if wellness ≥ 6."),
            ]
        else:
            days = [
                ("Quality speed / power", "high", 70, "Planned speed day: full recoveries between reps."),
                ("Recovery", "low", 35, "Easy aerobic + mobility. Protect 7.5h+ sleep."),
                ("Skill + strength", "moderate", 60, "Sport-specific skill work and a moderate strength circuit."),
            ]
        return [
            {"day": i + 1, "focus": f, "intensity": it, "minutes": m, "detail": d}
            for i, (f, it, m, d) in enumerate(days)
        ]


engine_singleton = AthleteLensModel()
