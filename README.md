# Athlete Lens

**Machine-learning framework for athlete performance analysis** — built for school, college and grassroots coaches in India who do **not** have GPS vests, force plates or a sports-science lab.

> A stopwatch, a tape, and how the session felt is enough.

**Live demo:** **https://athlete-lens.vercel.app** (Vercel, demo data resets on cold start) · API docs at `/docs`

![stack](https://img.shields.io/badge/FastAPI-0.115-009688) ![ml](https://img.shields.io/badge/scikit--learn-1.6-F7931E) ![deploy](https://img.shields.io/badge/deploy-Vercel%20%7C%20Docker-000) ![ci](https://github.com/sivaangayarkanni/athlete-lens/actions/workflows/ci.yml/badge.svg)

---

## Why this exists

District and college athletes in Tamil Nadu (and across India) already train hard. What they lack is a second pair of eyes that can turn a week of sessions into:

- a **performance index**
- a **readiness score** (green / amber / red)
- an **injury-risk band** (low / moderate / high) with a 7-day probability, broken down into **12 body regions** (hamstring, quadriceps, groin, calf/Achilles, knee, ankle, foot, hip, lower back, shoulder, elbow/wrist, head/neck incl. concussion), the likely injury type, the drivers behind it and what-if scenarios
- a coach-facing **plan for the next 72 hours**

Athlete Lens is that layer. It is intentionally *not* a computer-vision lab demo. It is a product a PE teacher can use after evening practice.

## What the product does

| Screen | What you get |
| --- | --- |
| **Dashboard** | Squad KPIs, injury-risk band split, **squad heatmap (athletes × 12 body regions)**, a watchlist with each athlete's top region, recent sessions, sport mix |
| **Roster** | Search / filter by sport and risk band, readiness ring per athlete, add athletes |
| **Athlete page** | **Interactive front/back SVG body map** coloured by risk vs the sport average; click a region for its drivers (Shapley values), risk trend, likely injury type, **model re-run what-ifs** ("cut load 15%: hamstring 0.26% → 0.22%"), prevention and return-to-play guidance; overall 7-day risk with drivers; injury-type mix; real-data cross-check; **load management** (daily sRPE, EWMA acute vs chronic, ACWR with the 0.8–1.3 band, Foster monotony and strain, daily risk trend for the top regions); injury log; readiness and 72-hour plan; session log |
| **Analysis lab** | What-if controls (this week's load change, extra rest days, matches, sleep, Nordic / Copenhagen programmes, injury history, growth) with a live body map. For a rostered athlete the model reruns on their logged calendar. Nothing is saved |
| **CSV import** | Bulk import from a hostel register / Excel export with per-row validation errors |
| **Model card** | Real vs simulated data split, per-region grouped-CV metrics vs baselines, oracle ceiling, calibration plots, top-k region accuracy, grouped permutation importance, real-data results, sources, limitations |

Sports: Athletics, Kabaddi, Kho-Kho, Football, Hockey, Volleyball, Badminton, Wrestling. Seed data includes eight Tamil Nadu athletes with eight weeks of daily sessions (matches, a camp load spike, an exam week with poor sleep, a youth growth spurt, a return from a shoulder sprain), so every screen has something to show.

## Architecture

```
Static UI (frontend/dist, vanilla JS + SVG)  —JSON—▶  FastAPI (backend/app/main.py)
Optional React + Recharts shell (frontend/src)          ├─ SQLAlchemy + SQLite (additive migrations)
                                                        ├─ backend/app/injury/   per-region injury engine
                                                        │    features.py  rolling athlete-season features (numpy)
                                                        │    simulate.py  literature-grounded season simulator
                                                        │    realdata.py  real runner logs (Lövdal 2021, CC0)
                                                        │    train.py     training + every reported metric
                                                        │    engine.py    serving, Shapley drivers, what-ifs
                                                        └─ ml_engine.py   GradientBoosting performance index + readiness / plan
```

Runtime needs only numpy + scikit-learn (no pandas, no SHAP / LightGBM) to keep the serverless bundle small. Models ship pre-trained in `backend/app/artifacts` (about 13 MB).

## Injury model: what is real and what is simulated

| Part | Data | Status |
| --- | --- | --- |
| 12 per-region models, overall model, injury-type mix | **Simulated**: 3,000 athlete-seasons (28-day burn-in + 168 days), 200,686 labelled athlete-days, 4,154 injuries (1.38 per athlete-season) | No open dataset with body-region labels **and** daily training load was found (the NFL playing-surface data needs a Kaggle login). The simulator is grounded in the published studies below. |
| Real-data cross-check | **Real**: Lövdal, den Hartigh & Azzopardi (2021), *Injury Prediction In Competitive Runners With Machine Learning*, DataverseNL [doi:10.34894/UWU9PV](https://doi.org/10.34894/UWU9PV), **CC0 1.0** (licence checked via DataCite). 74 runners, 42,798 athlete-days, 575 injury days | Binary labels (no body region). Downloaded from the [GitHub mirror](https://github.com/sonicjoy/Injury-Prediction-for-Competitive-Runners) (`week_approach_maskedID_timeseries.csv`, sha256 `373b01b1…b75d`) because DataverseNL's API sits behind an anti-bot wall. `python -m backend.app.injury.realdata` refetches it to `data/external/` (git-ignored). |
| Performance index | Synthetic cohort (unchanged from v1) | |

### Design

- **Label:** injury onset in region *r* within the next 7 days, on days the athlete is available.
- **Features** (68, computed per day with no look-ahead; the same code runs for training, the API and the simulator):
  - session-RPE load, EWMA acute (7-day) and chronic (28-day) load, EWMA and rolling ACWR, week-over-week load and distance change;
  - Foster monotony and strain, sessions, rest days, matches, hard (RPE ≥ 8) sessions, days since rest;
  - sleep mean, sleep debt (< 8 h) and last night, wellness mean and trend, resting-HR drift, sprint and jump drop vs the 28-day best, hop asymmetry;
  - injury history per region, days since the last injury and since return;
  - age, sex, BMI, training years, growth in the last 6 months (youth), Nordic and Copenhagen programme flags;
  - sport one-hot plus the sport's typical region mix.
- **Models:** one `HistGradientBoostingClassifier` per region plus one for "any injury", calibrated with `CalibratedClassifierCV` (sigmoid, athlete-grouped inner folds, `ensemble=False`). Trees are shallow (4 leaves, leaves ≥ 3% of rows, L2 = 10). Rows from one athlete are highly correlated (overlapping windows, constant profile), so deeper trees memorised which athletes got hurt and *lost* to the sport-rate baseline on unseen athletes. Shallow trees fixed that.
- **Validation:** 5-fold `GroupKFold` by athlete (each fold calibrates on 20% of its training athletes), plus a temporal split. The temporal split trains on 80% of athletes for the first two-thirds of the season and tests on the *other* 20% for the last third.
- **Injury type:** a regularised HGB multiclass model is compared with the region's type-frequency table by grouped-CV log loss, and the better one is shipped. In this run the frequency table won (log loss 0.933 vs 1.005), so the app uses it.
- **Explanations:** sampled Shapley values (24 antithetic permutations) over 12 feature groups. The reference is a median athlete of the same sport and sex, and contributions add up exactly to *p(athlete) − p(reference)*.
- **What-ifs:** the athlete's calendar is edited (−15% / −30% load, swap the hardest session for rest, 8 h sleep, skip the match, add Nordic / Copenhagen) and the features and models are re-run.

### Held-out metrics (from `python -m backend.app.injury.train`, all in `backend/app/artifacts/injury_metrics.json`)

Simulated cohort, 5-fold athlete-grouped CV (out-of-fold predictions):

| Target | Prevalence (7-day) | ROC AUC | Sport-rate AUC | PR AUC | Sport-rate PR AUC | Brier (constant) | Temporal AUC | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Any injury** | 6.37% | **0.642** | 0.520 | **0.1214** | 0.0674 | 0.05828 (0.05960) | 0.676 | 0.0024 |
| Hamstring | 0.66% | **0.791** | 0.697 | **0.0377** | 0.0124 | 0.00642 (0.00653) | 0.722 | 0.0009 |
| Quadriceps | 0.34% | **0.598** | 0.567 | **0.0050** | 0.0043 | 0.00341 (0.00341) | 0.593 | 0.0006 |
| Groin / adductor | 0.29% | **0.681** | 0.644 | **0.0082** | 0.0049 | 0.00284 (0.00284) | 0.688 | 0.0006 |
| Calf / Achilles | 0.44% | **0.642** | 0.583 | **0.0100** | 0.0051 | 0.00436 (0.00437) | 0.673 | 0.0009 |
| Knee (ACL/MCL, meniscus, patellar) | 1.04% | **0.667** | 0.512 | **0.0261** | 0.0103 | 0.01025 (0.01031) | 0.720 | 0.0009 |
| Ankle | 0.98% | **0.656** | 0.622 | **0.0206** | 0.0140 | 0.00969 (0.00973) | 0.656 | 0.0008 |
| Foot | 0.41% | **0.669** | 0.633 | **0.0102** | 0.0058 | 0.00410 (0.00411) | 0.704 | 0.0006 |
| Hip | 0.24% | **0.580** | 0.585 | **0.0032** | 0.0030 | 0.00237 (0.00237) | 0.534 | 0.0007 |
| Lower back | 0.50% | **0.620** | 0.555 | **0.0088** | 0.0059 | 0.00494 (0.00494) | 0.609 | 0.0007 |
| Shoulder | 0.62% | **0.769** | 0.721 | **0.0243** | 0.0119 | 0.00608 (0.00613) | 0.686 | 0.0009 |
| Elbow / wrist / hand | 0.42% | **0.600** | 0.604 | **0.0061** | 0.0054 | 0.00419 (0.00419) | 0.632 | 0.0007 |
| Head / neck (incl. concussion) | 0.43% | **0.715** | 0.716 | **0.0088** | 0.0083 | 0.00432 (0.00433) | 0.727 | 0.0011 |

| Any-injury model | ROC AUC | PR AUC | Brier |
| --- | --- | --- | --- |
| Calibrated HGB (shipped) | 0.642 | 0.1214 | 0.05828 |
| Constant prevalence | 0.500 | 0.0636 | 0.05960 |
| Sport injury rate | 0.520 | 0.0674 | 0.05959 |
| ACWR-only logistic | 0.569 | 0.1001 | 0.05901 |
| Logistic regression, all features | 0.638 | 0.1215 | 0.05831 |
| Oracle: simulator's true hazard (ceiling) | 0.707 | 0.2079 | 0.05560 |

- **Which region?** On windows that contain an injury, the top-1 region is right 22.6% of the time (sport prior 21.5%, global prior 16.4%). The top-3 regions contain it 49.9% of the time (sport prior 50.9%, global 42.1%).
- **Reading this honestly:** the overall model beats the sport rate and the ACWR-only model clearly, but only ties a well-regularised logistic regression. 9 of 12 regions beat the sport-rate baseline; hip, elbow/wrist and head/neck are within 0.005 of it. The oracle (the simulator's own true hazard, which knows each athlete's hidden frailty and future plan) reaches only 0.707, so most 7-day injury risk is irreducible noise even in a world we wrote ourselves. Every number measures how well the model recovers the simulator, **not real-world accuracy**.

Real runner data (Lövdal 2021):

| Real-data model (Lövdal 2021) | ROC AUC | PR AUC | Brier |
| --- | --- | --- | --- |
| 18 app-mappable features, athlete-grouped 5-fold CV | 0.566 | 0.0162 | 0.01326 |
| 18 app-mappable features, temporal split | 0.541 | 0.0192 | 0.01672 |
| All 70 dataset features, grouped CV (benchmark) | 0.622 | 0.0190 | 0.01327 |
| All 70 features, temporal split | 0.590 | 0.0252 | 0.01674 |
| Constant prevalence | 0.500 | 0.0134 | 0.01325 |

The real-data signal is weak (in line with the original paper's modest results). In this cohort, the injury rate after a big last-week km jump vs the 3-week average was *lower*, not higher (1.66% per day at a ratio < 0.8 vs 0.67% at ≥ 1.5). The simulator's load-spike effect comes from team-sport studies (Hulin 2016) and is not confirmed by these runners. The app shows this real model as a separate cross-check, not as the source of the region risks.

### Sources actually read (abstracts) and used

Each simulator effect is commented in `simulate.py` with its source, or marked "assumption" where the size was chosen by us.

- Foster 1998, PMID 9662690: session-RPE, monotony, strain
- Hulin 2016, PMID 26511006: ACWR spikes, protective chronic load
- Milewski 2014, PMID 25028798: < 8 h sleep, 1.7× injury
- Hägglund 2006, PMID 16855067: previous same-site injury
- Ekstrand 2011, PMID 19553225: match vs training incidence, thigh strains, re-injury
- López-Valenciano 2020, PMID 31171515: football injury epidemiology
- Petersen 2011, PMID 21825112: Nordic hamstring programme (RR 0.293)
- Harøy 2019, PMID 29891614: Copenhagen adductor programme (OR 0.59)
- Feddermann-Demont 2014, PMID 24620039: athletics injury distribution
- van Gent 2007, PMID 17473005: running injuries
- Fong 2007, PMID 17190537: ankle injuries by sport
- Young 2023, doi 10.1007/s12178-023-09826-2; Hammer 2020, doi 10.1177/2325967120903699; Johnson 2023, PMID 36922269: sport-specific distributions
- Tooth 2020, PMID 32758080: overhead / shoulder risk factors
- Lin 2018, PMID 29550413: sex differences
- Towlson 2021, PMID 32961300: growth / maturation in youth athletes

Performance index (GradientBoosting, synthetic cohort of 2,400, 20% hold-out): MAE **3.48** vs 9.53 for predict-the-mean, R² 0.858.

Retrain everything (about 4–5 minutes on 4 cores): `python -m backend.app.injury.train` (`--quick` for a smoke run). Performance model: `python -m backend.app.train`.

**This is not a medical device.** High risk means “talk to the physio / rest the block”, not a diagnosis.

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m uvicorn backend.app.main:app --reload --port 8000
```

Open `http://localhost:8000` (UI) or `http://localhost:8000/docs` (OpenAPI). The SQLite DB defaults to `/tmp/athlete_lens_v2.db` (older databases are migrated in place, additively); set `DATABASE_URL` to change it (see `.env.example`).

Optional React shell (EN / தமிழ் toggle):

```bash
cd frontend && npm install && npm run dev   # http://localhost:5173, proxies /api
```

## Deploy

### Vercel (serverless)

The repo deploys as-is with Vercel's zero-config FastAPI support: `index.py` exports the ASGI `app`, `vercel.json` trims the bundle, `.python-version` pins Python 3.12.

```bash
npm i -g vercel
vercel deploy --prod
```

On Vercel only `/tmp` is writable, so the SQLite DB lives in `/tmp` and the pre-trained models are loaded read-only from the bundle. **Data written on Vercel is ephemeral**: each function instance starts from the seeded demo roster on a cold start, and separate instances don't share writes. For persistent data point `DATABASE_URL` at a hosted Postgres (add a driver such as `psycopg[binary]` to `requirements.txt`).

### Docker (persistent SQLite volume)

```bash
docker compose up --build    # http://localhost:8000
```

## Tests

```bash
pytest -q
```

44 tests cover the API (CRUD, sessions, validation, what-if, CSV ingest, stats, model cards, static UI, per-region endpoints, heatmap, injury log, additive DB migration), the injury engine (rolling features incl. EWMA / ACWR / monotony / strain, no look-ahead, simulator determinism, calibrated outputs, Shapley additivity, what-if re-runs, real-data cross-check, recorded metrics beating baselines) and the performance model (feature building, training, artifact staleness). GitHub Actions runs them on every push / PR.

## API surface

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health`, `/api/health` | liveness + model meta |
| GET | `/api/stats` | squad KPIs, risk bands, watchlist, recent sessions |
| GET / POST | `/api/athletes` | list (`?sport=&risk=&q=`) / create |
| GET / PATCH / DELETE | `/api/athletes/{id}` | detail with timeline + latest analysis / update (rescores history) / delete |
| GET | `/api/athletes/{id}/sessions` | session list |
| POST | `/api/sessions` | log + score a session |
| DELETE | `/api/sessions/{id}` | delete a session |
| POST | `/api/analyze` | guest or what-if analysis incl. the per-region summary (`athlete_id` re-runs the injury model on that athlete's calendar; `load_change_pct`, `extra_rest_days`, `sleep_hours`, programme flags are what-if levers; guests get a synthetic 4-week history) |
| GET | `/api/athletes/{id}/injury-risk` | 7-day risk for all 12 regions, overall band, likely type, drivers (`?explain=none\|top\|all`), load snapshot, real-data cross-check |
| GET | `/api/athletes/{id}/injury-risk/{region}` | one region: drivers, what-if scenarios, daily trend, type mix, history, prevention / return-to-play guidance |
| GET | `/api/athletes/{id}/risk-trend` | daily overall + per-region risk, EWMA acute / chronic, ACWR, monotony, strain |
| GET | `/api/athletes/{id}/what-if` | counterfactual scenarios for the top regions |
| GET / POST | `/api/athletes/{id}/injuries` | injury log (adding / deleting one rescores the athlete's history) |
| DELETE | `/api/injuries/{id}` | delete an injury record |
| GET | `/api/squad/heatmap` | athletes × regions matrix (`?sport=`) |
| GET | `/api/regions`, `/api/injury-types` | region catalogue with prevention guidance, injury types |
| GET | `/api/injury-model` | full injury model card: every metric, calibration bins, importance, sources, limitations |
| POST | `/api/upload-csv` | bulk import (multipart `file`) |
| GET | `/api/model` | combined model card (performance + injury summary) |
| GET | `/api/sports` | supported sports |

Validation errors return `422` with a readable `detail` (`"age: Input should be greater than or equal to 10"`) plus a per-field `errors` list.

Sample file: [`data/sample_athletes.csv`](data/sample_athletes.csv)

## Product principles

1. **Works on a dust track.** No wearable lock-in.
2. **Explainable.** Every score ships with “why” and a next action.
3. **Bilingual-ready.** EN / TA toggle on the React shell.
4. **Honest ML.** Model card is a first-class screen, not a footnote.

## Author

Sivaangayarkanni Sathyan · B.Tech AI & Data Science
[github.com/sivaangayarkanni](https://github.com/sivaangayarkanni)

MIT licensed.
