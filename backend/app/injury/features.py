"""Rolling, athlete-season feature engineering (numpy only).

One function, `daily_features`, turns a contiguous calendar of daily training data for a
single athlete into a feature row for *every* day, using only data up to and including
that day (no look-ahead). The simulator, the trainer and the live API all call it, so
training and serving features are identical.

Load = session-RPE (duration_min x RPE, Foster 1998). Acute/chronic loads are computed both
as exponentially weighted moving averages (7- and 28-day spans) and as rolling sums
(7-day acute, 28-day weekly-average chronic). Monotony = 7-day mean / SD of daily load,
strain = 7-day load x monotony (Foster 1998).
"""
from __future__ import annotations

import numpy as np

from .catalog import REGIONS, SPORT_REGION_PRIOR, SPORTS

PROFILE_FEATURES = ["age", "female", "bmi", "years_training", "growth_cm", "youth_growth", "nordic", "adductor"]
LOAD_FEATURES = [
    "ewma_acute", "ewma_chronic", "acwr_ewma", "acute_7", "chronic_28w", "acwr_ra", "load_wow",
    "monotony_7", "strain_7", "sessions_7", "rest_days_7", "matches_7", "hi_rpe_7", "distance_7",
    "distance_wow", "days_since_rest",
]
RECOVERY_FEATURES = ["sleep_mean_7", "sleep_debt_7", "sleep_last", "wellness_mean_3", "wellness_trend",
                     "sprint_delta_pct", "jump_delta_pct", "rhr_delta", "asymmetry_pct"]
HISTORY_FEATURES = ["prev_any", "days_since_injury", "days_since_return"] + [f"prev_{r}" for r in REGIONS]
SPORT_FEATURES = [f"sport_{s}" for s in SPORTS]
# Each sport's typical share of injuries in each region (catalog priors). Lets a single tree split
# order sports by region-specific risk instead of needing several one-hot splits.
SPORT_PRIOR_FEATURES = [f"sport_prior_{r}" for r in REGIONS]
FEATURES = PROFILE_FEATURES + LOAD_FEATURES + RECOVERY_FEATURES + HISTORY_FEATURES + SPORT_FEATURES + SPORT_PRIOR_FEATURES

# Feature groups used for explanations and grouped permutation importance.
GROUPS = {
    "acute_load": ["ewma_acute", "acute_7", "sessions_7", "hi_rpe_7", "distance_7"],
    "load_spike_acwr": ["acwr_ewma", "acwr_ra", "load_wow", "distance_wow"],
    "chronic_fitness": ["ewma_chronic", "chronic_28w"],
    "monotony_strain": ["monotony_7", "strain_7", "rest_days_7", "days_since_rest"],
    "match_exposure": ["matches_7"],
    "sleep": ["sleep_mean_7", "sleep_debt_7", "sleep_last"],
    "wellness": ["wellness_mean_3", "wellness_trend", "rhr_delta"],
    "neuromuscular_fatigue": ["sprint_delta_pct", "jump_delta_pct"],
    "asymmetry": ["asymmetry_pct"],
    "injury_history": HISTORY_FEATURES,
    "age_growth": ["age", "growth_cm", "youth_growth", "years_training"],
    "sex_body": ["female", "bmi"],
    "prevention_programmes": ["nordic", "adductor"],
    "sport": SPORT_FEATURES + SPORT_PRIOR_FEATURES,
}
GROUP_LABELS = {
    "acute_load": "Acute training load",
    "load_spike_acwr": "Load spike / ACWR",
    "chronic_fitness": "Chronic load (fitness base)",
    "monotony_strain": "Monotony, strain & rest",
    "match_exposure": "Match exposure",
    "sleep": "Sleep",
    "wellness": "Wellness & resting HR",
    "neuromuscular_fatigue": "Sprint / jump fatigue",
    "asymmetry": "Limb asymmetry",
    "injury_history": "Injury history",
    "age_growth": "Age, growth & experience",
    "sex_body": "Sex & body size",
    "prevention_programmes": "Prevention programmes",
    "sport": "Sport",
}
assert sorted(sum(GROUPS.values(), [])) == sorted(FEATURES)
FIDX = {f: i for i, f in enumerate(FEATURES)}

DAILY_KEYS = ["load", "duration", "distance", "rpe", "sleep", "wellness", "sprint", "jump", "rhr",
              "asymmetry", "match", "session"]


def _ffill(a: np.ndarray, default: float) -> np.ndarray:
    out = a.astype(float).copy()
    last = default
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def _rolling_sum(a: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(a)])
    idx = np.arange(1, len(a) + 1)
    return c[idx] - c[np.maximum(0, idx - w)]


def _rolling_nanmean(a: np.ndarray, w: int) -> np.ndarray:
    v = np.where(np.isnan(a), 0.0, a)
    n = (~np.isnan(a)).astype(float)
    s, k = _rolling_sum(v, w), _rolling_sum(n, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(k > 0, s / np.maximum(k, 1e-9), np.nan)


def _ewma(a: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out = np.empty_like(a, dtype=float)
    m = 0.0
    for i, x in enumerate(a):
        m = alpha * x + (1 - alpha) * m
        out[i] = m
    return out


def _latest_vs_window(a: np.ndarray, w: int, mode: str) -> np.ndarray:
    """Latest observed value (within 7 days) relative to the best/mean of the preceding window."""
    out = np.full(len(a), np.nan)
    last_val, last_i = np.nan, -999
    for i in range(len(a)):
        if not np.isnan(a[i]):
            last_val, last_i = a[i], i
        if i - last_i > 7 or np.isnan(last_val):
            continue
        win = a[max(0, i - w + 1): i + 1]
        win = win[~np.isnan(win)]
        if len(win) < 3:
            continue
        ref = win.min() if mode == "min" else win.mean()
        out[i] = (last_val - ref) / ref * 100 if ref else np.nan
    return out


def daily_features(daily: dict[str, np.ndarray], profile: dict, injuries: list[tuple[int, str, int]] | None = None,
                   prior_regions: dict[str, int] | None = None) -> np.ndarray:
    """Return an (n_days x len(FEATURES)) matrix.

    daily: arrays of equal length (one entry per calendar day). Rest days have session=0,
           load/duration/distance=0 and NaN for rpe/sprint/jump/rhr/asymmetry. sleep/wellness
           may be NaN (forward-filled).
    profile: age, sex ('F'/'M'/...), height_cm, weight_kg, years_training, sport,
             growth_cm (cm grown in the last 6 months, optional), nordic, adductor (0/1).
    injuries: (onset_day_index, region, return_day_index) events inside the calendar.
    prior_regions: region -> count of injuries before the calendar starts.
    """
    n = len(daily["load"])
    injuries = injuries or []
    prior_regions = prior_regions or {}
    load = np.nan_to_num(daily["load"].astype(float))
    dist = np.nan_to_num(daily["distance"].astype(float))
    sess = np.nan_to_num(daily["session"].astype(float))
    match = np.nan_to_num(daily["match"].astype(float))
    rpe = daily["rpe"].astype(float)
    X = np.full((n, len(FEATURES)), np.nan)

    def put(name, val):
        X[:, FIDX[name]] = val

    # ---- profile
    female = 1.0 if str(profile.get("sex", "F")).upper().startswith("F") else 0.0
    age = float(profile["age"])
    growth = float(profile.get("growth_cm") or 0.0)
    put("age", age)
    put("female", female)
    put("bmi", float(profile["weight_kg"]) / (float(profile["height_cm"]) / 100) ** 2)
    put("years_training", float(profile.get("years_training", 2)))
    put("growth_cm", growth)
    put("youth_growth", growth if age <= 16 else 0.0)
    put("nordic", float(bool(profile.get("nordic"))))
    put("adductor", float(bool(profile.get("adductor"))))
    for s in SPORTS:
        put(f"sport_{s}", 1.0 if profile.get("sport") == s else 0.0)
    prior = SPORT_REGION_PRIOR.get(profile.get("sport"), SPORT_REGION_PRIOR["Athletics"])
    tot = sum(prior[r][0] if isinstance(prior[r], tuple) else prior[r] for r in REGIONS)
    for r in REGIONS:
        v = prior[r][0] if isinstance(prior[r], tuple) else prior[r]
        put(f"sport_prior_{r}", v / tot)

    # ---- load
    ea, ec = _ewma(load, 7), _ewma(load, 28)
    acute7 = _rolling_sum(load, 7)
    chronic28w = _rolling_sum(load, 28) / 4.0
    prev_week = np.concatenate([np.zeros(7), acute7[:-7]]) if n > 7 else np.zeros(n)
    d7 = _rolling_sum(dist, 7)
    prev_d7 = np.concatenate([np.zeros(7), d7[:-7]]) if n > 7 else np.zeros(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        put("ewma_acute", ea * 7)       # weekly-equivalent units
        put("ewma_chronic", ec * 7)
        put("acwr_ewma", np.where(ec > 1, ea / ec, np.nan))
        put("acute_7", acute7)
        put("chronic_28w", chronic28w)
        put("acwr_ra", np.where(chronic28w > 1, acute7 / chronic28w, np.nan))
        put("load_wow", np.where(prev_week > 1, np.clip(acute7 / prev_week - 1, -1, 3), np.nan))
        put("distance_7", d7)
        put("distance_wow", np.where(prev_d7 > 0.5, np.clip(d7 / prev_d7 - 1, -1, 3), np.nan))
        mean7 = acute7 / 7
        sq7 = _rolling_sum(load ** 2, 7) / 7
        sd7 = np.sqrt(np.maximum(sq7 - mean7 ** 2, 0))
        mono = np.where(mean7 > 0, mean7 / np.maximum(sd7, mean7 / 10), np.nan)  # cap at 10
        put("monotony_7", mono)
        put("strain_7", np.where(np.isnan(mono), 0, acute7 * mono))
    put("sessions_7", _rolling_sum(sess, 7))
    put("rest_days_7", 7 - _rolling_sum((sess > 0).astype(float), 7))
    put("matches_7", _rolling_sum(match, 7))
    put("hi_rpe_7", _rolling_sum(((np.nan_to_num(rpe) >= 8) & (sess > 0)).astype(float), 7))
    dsr = np.zeros(n)
    c = 0
    for i in range(n):
        c = c + 1 if sess[i] > 0 else 0
        dsr[i] = c
    put("days_since_rest", dsr)

    # ---- recovery / readiness
    sleep = _ffill(daily["sleep"], 7.5)
    well = _ffill(daily["wellness"], 7.0)
    put("sleep_mean_7", _rolling_nanmean(sleep, 7))
    put("sleep_debt_7", _rolling_sum(np.maximum(0, 8 - sleep), 7))   # hours below 8 h (Milewski 2014)
    put("sleep_last", sleep)
    w3 = _rolling_nanmean(well, 3)
    w14 = _rolling_nanmean(well, 14)
    put("wellness_mean_3", w3)
    put("wellness_trend", w3 - w14)
    put("sprint_delta_pct", _latest_vs_window(daily["sprint"].astype(float), 28, "min"))  # + = slower than best
    put("jump_delta_pct", _latest_vs_window(daily["jump"].astype(float), 28, "mean"))    # - = lower than usual
    rhr = daily["rhr"].astype(float)
    rhr_mean = _rolling_nanmean(rhr, 28)
    put("rhr_delta", _ffill(rhr, np.nan) - rhr_mean)
    put("asymmetry_pct", _ffill(daily["asymmetry"].astype(float), np.nan))

    # ---- injury history (as of each day)
    counts = {r: np.full(n, float(prior_regions.get(r, 0))) for r in REGIONS}
    since_inj = np.full(n, np.nan)
    since_ret = np.full(n, np.nan)
    for onset, region, ret in sorted(injuries):
        if region in counts and onset + 1 < n:
            counts[region][onset + 1:] += 1
        for i in range(max(onset + 1, 0), n):
            since_inj[i] = i - onset if np.isnan(since_inj[i]) else min(since_inj[i], i - onset)
        if ret is not None:
            for i in range(max(ret, 0), n):
                since_ret[i] = i - ret if np.isnan(since_ret[i]) else min(since_ret[i], i - ret)
    for r in REGIONS:
        put(f"prev_{r}", counts[r])
    put("prev_any", sum(counts.values()))
    put("days_since_injury", np.minimum(since_inj, 365))
    put("days_since_return", np.minimum(since_ret, 365))
    return X
