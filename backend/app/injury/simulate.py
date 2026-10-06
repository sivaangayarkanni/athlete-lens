"""Epidemiology-informed simulator of athlete seasons with per-region injuries.

Open data with injury *location* labels for these sports does not exist (that we could find),
so the per-region model is trained on simulated athlete-seasons. The generative hazard below
encodes published directions/magnitudes where we read them (see catalog.SOURCES) and explicit
assumptions elsewhere. The model never sees this formula, only the resulting daily logs, and
it is evaluated on athletes it never trained on. Its metrics therefore show how well it
recovers this simulator, not real-world accuracy.

Each athlete gets 28 burn-in days + `season_days` labelled days. Training phases, camps
(load spikes), deloads, exam weeks (less sleep), matches, rest days, injuries with
type-specific layoffs and (sometimes rushed) return-to-play are all simulated day by day.
"""
from __future__ import annotations

import numpy as np

from .catalog import (CONTACT_SPORTS, OVERHEAD_SPORTS, REGION_TYPES, REGIONS, RUNNING_SPORTS, SPORT_REGION_PRIOR,
                      SPORTS, TEAM_SPORTS, INJURY_TYPES)
from .features import daily_features

BURN_IN = 28
RIDX = {r: i for i, r in enumerate(REGIONS)}
KM_PER_MIN = {"Athletics": 0.11, "Football": 0.075, "Hockey": 0.07, "Kho-Kho": 0.05, "Kabaddi": 0.025,
              "Volleyball": 0.015, "Badminton": 0.02, "Wrestling": 0.01}
OVERUSE_REGIONS = {"calf_achilles", "foot", "lower_back", "knee", "shoulder", "groin", "hip"}
MUSCLE_REGIONS = {"hamstring", "quadriceps", "groin", "calf_achilles", "hip"}
LOWER_LIMB = {"hamstring", "quadriceps", "groin", "calf_achilles", "knee", "ankle", "foot", "hip"}
BASE_DAILY_HAZARD = 0.0034   # any-region hazard on a normal training day for an average athlete (tuned assumption)
MATCH_EXPOSURE = 3.5         # competition >> practice (Ekstrand 2011: 27.5 vs 4.1/1000 h; Hammer 2020: RR 4.0)
REST_EXPOSURE = 0.12
# Risk modifiers inflate some regions more than others (e.g. knee gets sex, growth, history and
# overuse effects). These factors rescale the base so simulated region shares stay close to the
# sport priors in catalog.SPORT_REGION_PRIOR. Derived from one 400-athlete calibration run.
REGION_CALIBRATION = dict(hamstring=1.1, quadriceps=1.57, groin=0.93, calf_achilles=0.85, knee=0.63, ankle=1.4,
                          foot=1.0, hip=1.1, lower_back=1.15, shoulder=1.0, elbow_wrist=1.44, head_neck=0.9)


def _prior(sport: str) -> np.ndarray:
    p = np.array([SPORT_REGION_PRIOR[sport][r] for r in REGIONS], dtype=float)
    return p / p.sum()


def sample_profile(rng: np.random.Generator, sport: str | None = None) -> dict:
    sport = sport or str(rng.choice(SPORTS))
    sex = "F" if rng.random() < 0.45 else "M"
    age = int(np.clip(round(rng.normal(19, 3.6)), 13, 30))
    height = float(np.clip(rng.normal(160 if sex == "F" else 173, 7), 140, 200))
    weight = float(np.clip(rng.normal(53 if sex == "F" else 66, 8), 36, 110))
    years = float(np.clip(rng.normal(max(1, age - 13), 2), 0.5, 18))
    growth = float(np.clip(rng.gamma(2.0, 1.6), 0, 10)) if age <= 16 else float(np.clip(rng.normal(0.3, 0.4), 0, 2))
    football_like = sport in {"Football", "Hockey"}
    return dict(
        sport=sport, sex=sex, age=age, height_cm=height, weight_kg=weight, years_training=years, growth_cm=growth,
        nordic=int(rng.random() < (0.35 if football_like else 0.15)),
        adductor=int(rng.random() < (0.30 if football_like else 0.10)),
    )


def simulate_athlete(rng: np.random.Generator, profile: dict, season_days: int = 168, allow_injuries: bool = True,
                     plan: dict | None = None) -> dict:
    """Simulate one athlete. Returns daily arrays, injuries [(onset, region, return_day, type)], prior regions."""
    plan = plan or {}
    sport = profile["sport"]
    D = BURN_IN + season_days
    female = profile["sex"] == "F"
    age = profile["age"]
    prior = _prior(sport)

    # latent athlete traits
    capacity = float(plan.get("capacity", np.exp(rng.normal(np.log(2300), 0.3))))       # weekly sRPE AU
    spw = int(plan.get("sessions_per_week", rng.choice([4, 5, 5, 6, 6, 7])))
    sleep_habit = float(plan.get("sleep_habit", np.clip(rng.normal(7.2, 0.6), 5.4, 8.9)))
    asym_base = float(plan.get("asymmetry", np.clip(abs(rng.normal(6, 4)), 1, 25)))
    frailty = float(np.exp(rng.normal(0, 0.35)))
    rush = rng.random() < 0.3
    n_prior = int(plan.get("n_prior", rng.poisson(0.8 * min(1.0, profile["years_training"] / 4))))
    prior_regions: dict[str, int] = dict(plan.get("prior_regions", {}))
    if "prior_regions" not in plan:
        for _ in range(n_prior):
            r = REGIONS[int(rng.choice(len(REGIONS), p=prior))]
            prior_regions[r] = prior_regions.get(r, 0) + 1
    sprint_base = (14.2 if female else 12.9) - 0.04 * min(profile["years_training"], 10) + rng.normal(0, 0.4)
    jump_base = (38 if female else 48) + rng.normal(0, 5)
    rhr_base = (64 if female else 60) + rng.normal(0, 4)

    # weekly plan
    n_weeks = D // 7 + 1
    week_mult = np.ones(n_weeks)
    week_sleep = np.zeros(n_weeks)
    for w in range(n_weeks):
        ramp = 0.65 + 0.35 * min(1.0, w / 5)
        u = rng.random()
        if u < plan.get("p_camp", 0.10):
            m = rng.uniform(1.35, 1.85); week_sleep[w] = -0.4
        elif u < plan.get("p_camp", 0.10) + 0.08:
            m = 0.6
        elif u < plan.get("p_camp", 0.10) + 0.14:
            m = 0.75; week_sleep[w] = -1.0           # exam week
        else:
            m = rng.normal(1.0, 0.1)
        week_mult[w] = ramp * max(0.3, m)
    for w, m in plan.get("week_overrides", {}).items():   # used by the demo seeder
        week_mult[w if w >= 0 else n_weeks + w] = m
    for w, s in plan.get("sleep_overrides", {}).items():
        week_sleep[w if w >= 0 else n_weeks + w] = s

    keys = ["load", "duration", "distance", "rpe", "sleep", "wellness", "sprint", "jump", "rhr", "asymmetry", "match", "session"]
    daily = {k: np.full(D, np.nan) for k in keys}
    for k in ("load", "duration", "distance", "match", "session"):
        daily[k][:] = 0.0
    hazard = np.full(D, np.nan)   # true simulated hazard (never a model feature; used only for the oracle ceiling)
    injured_until = -1
    ramp_start = -999
    injuries: list[tuple[int, str, int, str]] = []
    recent_region: dict[str, int] = {}
    ea = ec = 0.0
    a7, c28 = 2 / 8, 2 / 29

    for d in range(D):
        w, dow = d // 7, d % 7
        if dow == 0:   # choose rest days for the week
            rest = set(rng.choice(7, size=7 - spw, replace=False).tolist()) if spw < 7 else set()
            comp_week = w >= plan.get("matches_from_week", 5)
            match_day = 5 if (sport in TEAM_SPORTS and comp_week and rng.random() < 0.8) or \
                             (sport not in TEAM_SPORTS and comp_week and rng.random() < 0.3) else None
            rest.discard(match_day) if match_day is not None else None
        injured = d <= injured_until
        is_match = (dow == match_day) and not injured
        train = (dow not in rest or is_match) and not injured
        # sleep / wellness are recorded every day (also on rest days)
        sleep = float(np.clip(sleep_habit + week_sleep[w] + rng.normal(0, 0.6) - (0.8 if d > 0 and daily["match"][d - 1] else 0), 3.5, 10))
        daily["sleep"][d] = sleep
        load = 0.0
        if train:
            target = capacity * week_mult[w] / spw
            if d - ramp_start < 14 and not rush:
                target *= 0.5 + 0.5 * (d - ramp_start) / 14
            if is_match:
                rpe = float(np.clip(rng.normal(8.4, 0.6), 6, 10)); dur = float(np.clip(rng.normal(80, 12), 40, 120))
            else:
                rpe = float(np.clip(rng.normal(6.0, 1.3), 2, 10))
                dur = float(np.clip(target * np.exp(rng.normal(0, 0.3)) / rpe, 20, 180))
            load = rpe * dur
            daily["session"][d] = 1; daily["match"][d] = float(is_match)
            daily["rpe"][d] = round(rpe, 1); daily["duration"][d] = round(dur)
            daily["distance"][d] = round(dur * KM_PER_MIN[sport] * np.exp(rng.normal(0, 0.15)), 1)
        daily["load"][d] = load
        ea = a7 * load + (1 - a7) * ea
        ec = c28 * load + (1 - c28) * ec
        acwr = ea / ec if ec > 1 else 1.0
        fatigue = max(-0.5, min(1.5, acwr - 1))
        lo = max(0, d - 6)
        sleep7 = float(np.mean(daily["sleep"][lo:d + 1]))
        wellness = float(np.clip(7.6 - 2.2 * max(0, acwr - 1.1) - 0.7 * (7.5 - sleep7) + rng.normal(0, 0.7)
                                 - (2.5 if injured else 0), 1, 10))
        daily["wellness"][d] = round(wellness, 1)
        if train:
            daily["rhr"][d] = round(rhr_base + 4 * fatigue + 1.2 * (7.5 - sleep) + rng.normal(0, 1.5))
            if rng.random() < 0.25:
                daily["sprint"][d] = round(sprint_base * (1 + 0.025 * fatigue) + rng.normal(0, 0.08), 2)
            if rng.random() < 0.3:
                daily["jump"][d] = round(jump_base * (1 - 0.07 * fatigue) + rng.normal(0, 1.0), 1)
            if dow == 1 or rng.random() < 0.1:
                recent_ll = any(r in LOWER_LIMB and d - t < 35 for r, t in recent_region.items())
                daily["asymmetry"][d] = round(max(0, asym_base + 3 * fatigue + (5 if recent_ll else 0) + rng.normal(0, 2)), 1)

        if not allow_injuries or injured or d < 7:
            continue

        # ------------------------------------------------------------------ hazard
        exposure = MATCH_EXPOSURE if is_match else (1.0 if train else REST_EXPOSURE)
        sess7 = daily["session"][lo:d + 1].sum()
        loads7 = daily["load"][lo:d + 1]
        mono = loads7.mean() / max(loads7.std(), loads7.mean() / 10) if loads7.mean() > 0 else 0
        well3 = float(np.mean(daily["wellness"][max(0, d - 2):d + 1]))
        km7 = daily["distance"][lo:d + 1].sum()
        km28w = daily["distance"][max(0, d - 27):d + 1].sum() / 4
        jumps = daily["jump"][max(0, d - 27):d + 1]
        jumps = jumps[~np.isnan(jumps)]
        jump_drop = (jumps[-1] - jumps.mean()) / jumps.mean() if len(jumps) >= 3 else 0.0
        sprints = daily["sprint"][max(0, d - 27):d + 1]
        sprints = sprints[~np.isnan(sprints)]
        sprint_slow = (sprints[-1] - sprints.min()) / sprints.min() if len(sprints) >= 3 else 0.0
        asym = np.nanmax([asym_base, *(daily["asymmetry"][lo:d + 1][~np.isnan(daily["asymmetry"][lo:d + 1])])])
        hi_rpe = int(((np.nan_to_num(daily["rpe"][lo:d + 1]) >= 8)).sum())
        dsr = 0
        for k in range(d, -1, -1):
            if daily["session"][k] > 0: dsr += 1
            else: break

        spike = max(0.0, acwr - 1.3)                                     # Hulin 2016: spikes >~1.5 riskiest
        common = frailty * exposure
        common *= 1 + 0.7 * np.clip((8 - sleep7) / 1.5, 0, 1.3)          # Milewski 2014: <8 h sleep ~1.7x
        if well3 < 5: common *= 1.3                                      # assumption
        if d >= BURN_IN and acwr < 0.8 and train: common *= 1.15         # assumption (under-preparedness)
        if ec * 7 >= 0.9 * capacity: common *= 0.85                      # Hulin 2016: high chronic load protective
        if sum(prior_regions.values()) + len(injuries) > 0: common *= 1.3  # Hägglund 2006: previous injury HR 2.7 (part)
        h = np.zeros(len(REGIONS))
        for i, r in enumerate(REGIONS):
            f = prior[i] * REGION_CALIBRATION[r]
            f *= 1 + (0.5 if r in {"head_neck", "elbow_wrist"} else 2.0) * spike
            if r in OVERUSE_REGIONS:
                if mono > 2.0: f *= 1 + 0.4 * min(1.5, mono - 2.0)       # Foster 1998: monotony / strain
                if dsr >= 10: f *= 1.25                                   # assumption
            prev = prior_regions.get(r, 0) + sum(1 for inj in injuries if inj[1] == r)
            if prev:
                f *= 2.5 if r in {"hamstring", "groin", "knee"} else (1.0 if r == "ankle" else 1.6)  # Hägglund 2006
            if r in recent_region and d - recent_region[r] < 42:
                f *= 1.8                                                  # early re-injury (Ekstrand 2011: 12% re-injuries)
            if female:
                f *= {"knee": 1.6, "foot": 1.4, "calf_achilles": 1.2, "head_neck": 1.2}.get(r, 1.0)  # Lin 2018
            if age <= 16 and r in {"knee", "lower_back", "calf_achilles", "foot"}:
                f *= min(1.5, 1 + (0.08 if r != "foot" else 0.04) * profile["growth_cm"])   # Towlson 2021 (magnitude assumed)
            if asym > 10 and r in {"hamstring", "knee", "ankle"}:
                f *= min(1.8, 1 + (0.04 if r != "ankle" else 0.02) * (asym - 10))  # assumption
            if age > 24 and r in {"hamstring", "calf_achilles"}: f *= 1.25          # assumption
            if jump_drop < -0.06 and r in {"hamstring", "knee", "quadriceps"}:
                f *= 1.4 if r == "hamstring" else 1.2                                # assumption
            if sprint_slow > 0.03 and r == "hamstring": f *= 1.3                     # assumption
            if sport in RUNNING_SPORTS and km28w > 3 and km7 / km28w > 1.4 and r in {"calf_achilles", "foot"}:
                f *= 1.5                                                             # assumption
            if not female and km7 > 45 and r in {"calf_achilles", "foot", "knee"}: f *= 1.3   # van Gent 2007
            if hi_rpe >= 3 and r in {"hamstring", "groin", "quadriceps"}: f *= 1.2   # assumption
            if is_match and sport in CONTACT_SPORTS:
                f *= {"head_neck": 2.0, "shoulder": 1.5, "elbow_wrist": 1.3}.get(r, 1.0)  # assumption (contact)
            if sport in OVERHEAD_SPORTS and r == "shoulder": f *= 1 + 0.15 * max(0, sess7 - 4)  # Tooth 2020
            if r == "hamstring" and profile.get("nordic"): f *= 0.293                # Petersen 2011
            if r == "groin" and profile.get("adductor"): f *= 0.59                   # Harøy 2019
            h[i] = BASE_DAILY_HAZARD * common * f * len(REGIONS) / 12
        hazard[d] = float(h.sum())
        p = 1 - np.exp(-h)
        hits = np.nonzero(rng.random(len(REGIONS)) < p)[0]
        if len(hits):
            i = int(rng.choice(hits)); r = REGIONS[i]
            weights = dict(REGION_TYPES[r])
            if is_match and sport in CONTACT_SPORTS:
                for t, m in (("contusion", 2.0), ("ligament_sprain", 1.5), ("concussion", 2.0)):
                    if t in weights: weights[t] *= m
            if "overuse_stress" in weights and (mono > 2 or (km28w > 3 and km7 / km28w > 1.4) or (female and r == "foot")):
                weights["overuse_stress"] *= 2.0
            if "tendinopathy" in weights and ec * 7 > capacity and sess7 >= 6: weights["tendinopathy"] *= 1.5
            if "muscle_strain" in weights and (spike > 0.2 or jump_drop < -0.06): weights["muscle_strain"] *= 1.5
            types, wts = list(weights), np.array(list(weights.values()))
            itype = types[int(rng.choice(len(types), p=wts / wts.sum()))]
            lay = {"muscle_strain": (10, 28), "ligament_sprain": (7, 35), "tendinopathy": (10, 30),
                   "overuse_stress": (21, 49), "contusion": (2, 8), "concussion": (8, 21)}[itype]
            layoff = int(rng.integers(*lay))
            if itype == "ligament_sprain" and r == "knee" and rng.random() < 0.15:
                layoff = 150
            injuries.append((d, r, d + layoff + 1, itype))
            recent_region[r] = d + layoff
            injured_until = d + layoff
            ramp_start = d + layoff + 1
    return {"daily": daily, "injuries": injuries, "prior_regions": prior_regions, "profile": profile, "hazard": hazard}


def athlete_rows(sim: dict, horizon: int = 7, stride: int = 1):
    """Feature rows + 7-day-ahead region labels for labelled days where the athlete is available."""
    daily, inj = sim["daily"], sim["injuries"]
    D = len(daily["load"])
    X = daily_features(daily, sim["profile"], [(o, r, ret) for o, r, ret, _ in inj], sim["prior_regions"])
    unavailable = np.zeros(D, bool)
    onset_region = np.full(D, -1)
    for o, r, ret, _ in inj:
        unavailable[o:min(D, ret)] = True
        onset_region[o] = RIDX[r]
    days = [t for t in range(BURN_IN, D - horizon, stride) if not unavailable[t]]
    Y = np.zeros((len(days), len(REGIONS)), dtype=np.int8)
    for k, t in enumerate(days):
        for j in range(t + 1, t + horizon + 1):
            if onset_region[j] >= 0:
                Y[k, onset_region[j]] = 1
    # Oracle: mean TRUE daily hazard over the at-risk days of the window x horizon. Days after an
    # onset carry no hazard (athlete is out), so a plain sum would be censored by the outcome itself.
    hz = sim.get("hazard", np.full(D, np.nan))
    oracle = np.array([horizon * np.nan_to_num(np.nanmean(w)) if np.isfinite(w).any() else 0.0
                       for w in (hz[t + 1:t + horizon + 1] for t in days)])
    return X[days], Y, np.array(days), oracle


def simulate_cohort(n_athletes: int, seed: int = 7, season_days: int = 168, stride: int = 1):
    rng = np.random.default_rng(seed)
    Xs, Ys, groups, days, sports, oracles = [], [], [], [], [], []
    events = []   # (feature row the day before onset, region index, type index, athlete id)
    n_inj = 0
    for a in range(n_athletes):
        sim = simulate_athlete(rng, sample_profile(rng), season_days=season_days)
        X, Y, d, o = athlete_rows(sim, stride=stride)
        oracles.append(o)
        Xs.append(X); Ys.append(Y); groups.append(np.full(len(d), a)); days.append(d)
        sports.append(np.full(len(d), SPORTS.index(sim["profile"]["sport"])))
        if sim["injuries"]:
            Xfull = daily_features(sim["daily"], sim["profile"], [(o, r, ret) for o, r, ret, _ in sim["injuries"]], sim["prior_regions"])
            for o, r, _, t in sim["injuries"]:
                if o >= BURN_IN:
                    n_inj += 1
                    events.append((Xfull[o - 1], RIDX[r], INJURY_TYPES.index(t), a))
    return {
        "X": np.vstack(Xs), "Y": np.vstack(Ys), "groups": np.concatenate(groups), "days": np.concatenate(days),
        "sport": np.concatenate(sports), "oracle": np.concatenate(oracles), "events": events, "n_injuries": n_inj, "n_athletes": n_athletes,
        "season_days": season_days,
    }
