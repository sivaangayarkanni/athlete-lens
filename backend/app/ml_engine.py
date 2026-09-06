from __future__ import annotations
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import settings

FEATURE_NUM = ["age","years_training","height_cm","weight_kg","bmi","previous_injuries","duration_min","distance_km","sprint_100m_s","vertical_jump_cm","resting_hr","session_hr_avg","hr_reserve_proxy","rpe","sleep_hours","wellness","sessions_last_7","rest_days_last_7","acute_load","is_female"]
FEATURE_CAT = ["sport"]
SPORTS = ["Athletics","Kabaddi","Kho-Kho","Football","Hockey","Volleyball","Badminton","Wrestling"]

def enrich(row: dict) -> dict:
    out = dict(row)
    out["bmi"] = float(out["weight_kg"]) / ((float(out["height_cm"]) / 100) ** 2)
    out["hr_reserve_proxy"] = float(out["session_hr_avg"]) - float(out["resting_hr"])
    out["acute_load"] = float(out["duration_min"]) * float(out["rpe"]) * max(1, int(out["sessions_last_7"])) / 7.0
    out["is_female"] = 1.0 if str(out.get("sex","F")).upper().startswith("F") else 0.0
    if out.get("sport") not in SPORTS:
        out["sport"] = "Athletics"
    return out

def _cohort(n=1800, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        sex = rng.choice(["F","M"])
        sport = rng.choice(SPORTS)
        age = int(rng.integers(13,28))
        years = float(max(0.3, rng.normal(age-14, 1.8)))
        height = float(np.clip(rng.normal(160 if sex=="F" else 172, 8), 145, 198))
        weight = float(np.clip(rng.normal(52 if sex=="F" else 64, 8), 38, 105))
        sprint = float(np.clip(rng.normal(14.4 if sex=="F" else 13.1, 1.1), 10.6, 18.5))
        jump = float(np.clip(rng.normal(38 if sex=="F" else 46, 8), 22, 78))
        rhr = float(np.clip(rng.normal(66 if sex=="F" else 62, 7), 44, 92))
        sessions7 = int(np.clip(rng.integers(2,9),1,12))
        rest7 = int(np.clip(7-sessions7+int(rng.integers(-1,2)),0,6))
        duration = float(np.clip(rng.normal(70,18),25,160))
        distance = float(np.clip(rng.normal(5.2,2.4),0.2,22))
        rpe = float(np.clip(rng.normal(6.4,1.5),2,10))
        sleep = float(np.clip(rng.normal(6.8,1.1),3.5,10.5))
        wellness = float(np.clip(rng.normal(6.9,1.4),2,10))
        inj = int(np.clip(rng.poisson(0.6),0,6))
        shr = float(np.clip(rhr+rng.normal(78,14),95,195))
        row = enrich(dict(age=age,sex=sex,sport=sport,years_training=years,height_cm=height,weight_kg=weight,previous_injuries=inj,duration_min=duration,distance_km=distance,sprint_100m_s=sprint,vertical_jump_cm=jump,resting_hr=rhr,session_hr_avg=shr,rpe=rpe,sleep_hours=sleep,wellness=wellness,sessions_last_7=sessions7,rest_days_last_7=rest7))
        speed = np.clip((18-sprint)/7.5,0,1); power=np.clip((jump-20)/50,0,1)
        fatigue=np.clip((row["acute_load"]-180)/280,0,1); sleep_pen=np.clip((7.2-sleep)/4,0,1)
        perf = 100*np.clip(0.28*speed+0.2*power+0.16*np.clip((75-rhr)/40,0,1)+0.12*(wellness/10)+0.1*np.clip(years/8,0,1)-0.18*fatigue-0.1*sleep_pen+rng.normal(0,0.04),0.12,0.97)
        logit = -2.2+0.55*inj+1.1*fatigue+0.7*sleep_pen+0.35*(rpe>=8)+0.45*(rest7<=1)+0.25*(sessions7>=7)+rng.normal(0,0.35)
        p = 1/(1+np.exp(-logit))
        row["performance_index"]=float(round(perf,2)); row["injury_label"]=int(rng.random()<p)
        rows.append(row)
    return pd.DataFrame(rows)

class AthleteLensModel:
    def __init__(self):
        self.dir = Path(settings.model_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.clf_path=self.dir/"injury_rf.joblib"; self.reg_path=self.dir/"performance_gbr.joblib"; self.meta_path=self.dir/"meta.joblib"
        self.clf=None; self.reg=None; self.meta={}
    def _pipe(self, est):
        pre = ColumnTransformer([("num", StandardScaler(), FEATURE_NUM),("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_CAT)])
        return Pipeline([("pre", pre),("model", est)])
    def train(self, n=1800):
        df=_cohort(n); X=df[FEATURE_NUM+FEATURE_CAT]
        self.clf=self._pipe(RandomForestClassifier(n_estimators=140,max_depth=9,min_samples_leaf=8,class_weight="balanced",random_state=42))
        self.reg=self._pipe(GradientBoostingRegressor(n_estimators=120,max_depth=3,learning_rate=0.08,random_state=42))
        self.clf.fit(X, df["injury_label"]); self.reg.fit(X, df["performance_index"])
        acc=float((self.clf.predict(X)==df["injury_label"]).mean())
        mae=float(np.mean(np.abs(self.reg.predict(X)-df["performance_index"])))
        self.meta={"trained_on":int(n),"injury_train_accuracy":round(acc,3),"performance_train_mae":round(mae,2),"sports":SPORTS,"features":FEATURE_NUM+FEATURE_CAT}
        joblib.dump(self.clf,self.clf_path); joblib.dump(self.reg,self.reg_path); joblib.dump(self.meta,self.meta_path)
        return self.meta
    def load(self):
        if self.clf_path.exists() and self.reg_path.exists():
            self.clf=joblib.load(self.clf_path); self.reg=joblib.load(self.reg_path)
            self.meta=joblib.load(self.meta_path) if self.meta_path.exists() else {}
        else:
            self.train()
    def _frame(self, payload):
        row=enrich(payload)
        return pd.DataFrame([{k:row.get(k) for k in FEATURE_NUM+FEATURE_CAT}])
    def analyze(self, payload):
        if self.clf is None or self.reg is None: self.load()
        row=enrich(payload); X=self._frame(payload)
        p_inj=float(self.clf.predict_proba(X)[0,1]); perf=float(np.clip(self.reg.predict(X)[0],8,98))
        load=float(np.clip(row["acute_load"]/3.2,0,100))
        recovery=float(np.clip(12*row["sleep_hours"]+6*row["wellness"]+8*row["rest_days_last_7"]-4*max(0,row["rpe"]-6)-0.08*row["acute_load"],8,98))
        speed=float(np.clip((18.2-row["sprint_100m_s"])/7.4*100,5,99))
        power=float(np.clip((row["vertical_jump_cm"]-18)/52*100,5,99))
        readiness=float(np.clip(0.42*recovery+0.22*(100-min(load,100))+0.18*perf+0.18*row["wellness"]*10,6,98))
        if p_inj>=0.55 or (row["rest_days_last_7"]<=1 and row["rpe"]>=8): risk="high"
        elif p_inj>=0.32: risk="moderate"
        else: risk="low"
        overtraining=bool(row["sessions_last_7"]>=7 and row["sleep_hours"]<6.5 and row["rpe"]>=7.5)
        return {
            "performance_index":round(perf,1),"readiness_score":round(readiness,1),"injury_risk":risk,
            "injury_probability":round(p_inj,3),"overtraining":overtraining,"load_score":round(min(load,100),1),
            "recovery_score":round(recovery,1),"speed_score":round(speed,1),"power_score":round(power,1),
            "explanations":self._explain(row,p_inj,load,recovery),"recommendations":self._recs(row,risk,overtraining),
            "feature_importance":self._importance(),
        }
    def _importance(self):
        try:
            model=self.clf.named_steps["model"]; pre=self.clf.named_steps["pre"]
            pairs=sorted(zip(pre.get_feature_names_out(), model.feature_importances_), key=lambda x:-x[1])[:8]
            return [{"feature":n.split("__")[-1].replace("_"," "),"weight":round(float(v),3)} for n,v in pairs]
        except Exception:
            return [{"feature":"acute load","weight":0.18},{"feature":"previous injuries","weight":0.15},{"feature":"sleep hours","weight":0.13}]
    def _explain(self, row, p_inj, load, recovery):
        notes=[]
        if row["sleep_hours"]<6.5: notes.append("Sleep is below the 7-hour recovery line.")
        if row["rest_days_last_7"]<=1: notes.append("Almost no rest day in the last week.")
        if row["rpe"]>=8: notes.append("Session felt very hard (RPE >= 8).")
        if row["previous_injuries"]>=2: notes.append("Injury history is non-zero — progress load slowly.")
        if row["acute_load"]>280: notes.append("Acute training load is spiked. Cut volume 20-30% for 4-5 days.")
        if not notes: notes.append("Load, sleep and wellness are balanced.")
        notes.append(f"Model injury probability is {p_inj:.0%} with load {load:.0f}/100 and recovery {recovery:.0f}/100.")
        return notes[:5]
    def _recs(self, row, risk, overtraining):
        recs=[]
        if risk=="high" or overtraining:
            recs.append({"title":"Deload this block","detail":"Drop high-intensity reps. Keep mobility + easy aerobic 25-35 min.","priority":"now","tag":"injury"})
        elif risk=="moderate":
            recs.append({"title":"Cap intensity","detail":"One quality speed or power session only.","priority":"this_week","tag":"load"})
        else:
            recs.append({"title":"Green light for quality work","detail":"Keep the planned speed day. Protect sleep at 7.5h+ after it.","priority":"this_week","tag":"performance"})
        if row["sleep_hours"]<7:
            recs.append({"title":"Fix the night before the session","detail":"Target 7.5 hours. A 20-min walk after dinner helps more than extra drills.","priority":"now","tag":"recovery"})
        recs.append({"title":"Weekly test you can do on any ground","detail":"Record 30m fly, standing long jump, and wellness (1-10).","priority":"monitor","tag":"test"})
        return recs[:4]

engine_singleton = AthleteLensModel()
