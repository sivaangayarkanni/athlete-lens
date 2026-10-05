import { useEffect, useMemo, useState } from "react";
import {
  Bar, CartesianGrid, ComposedChart, Legend, Line, LineChart, ReferenceArea,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "./api";

// Minimal EN / TA dictionary for the coach-facing labels.
const T = {
  en: { dashboard: "Dashboard", roster: "Roster", lab: "Analysis lab", athletes: "Athletes", sessions: "Sessions",
        readiness: "Readiness", highRisk: "High risk", performance: "Performance", risk: "Injury risk", plan: "Next 72 hours",
        load: "Training load", why: "Why", recs: "Recommendations", watch: "Watchlist", select: "Select an athlete" },
  ta: { dashboard: "முகப்பு", roster: "வீரர்கள்", lab: "பகுப்பாய்வு", athletes: "வீரர்கள்", sessions: "பயிற்சிகள்",
        readiness: "தயார்நிலை", highRisk: "அதிக ஆபத்து", performance: "செயல்திறன்", risk: "காய ஆபத்து", plan: "அடுத்த 72 மணி நேரம்",
        load: "பயிற்சி சுமை", why: "ஏன்", recs: "பரிந்துரைகள்", watch: "கவனிப்பு பட்டியல்", select: "ஒரு வீரரைத் தேர்ந்தெடுக்கவும்" },
};
const RISK = { low: "text-mint", moderate: "text-ember", high: "text-red-400" };
const SLIDERS = [
  ["sleep_hours", 3, 12, 0.1], ["rpe", 1, 10, 0.5], ["duration_min", 10, 300, 5], ["wellness", 1, 10, 0.5],
  ["sessions_last_7", 0, 14, 1], ["rest_days_last_7", 0, 7, 1],
];

const Card = ({ children, className = "" }) => <div className={`glass rounded-2xl p-5 ${className}`}>{children}</div>;
const Kpi = ({ label, value, tone = "" }) => (
  <Card><p className="text-xs uppercase tracking-wider text-slate-400">{label}</p><p className={`mt-1 font-display text-3xl ${tone}`}>{value}</p></Card>
);

function Analysis({ a, t }) {
  if (!a) return null;
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card>
        <div className="flex gap-6">
          <div><p className="text-xs text-slate-400">{t.readiness}</p><p className="font-display text-4xl">{Math.round(a.readiness_score)}</p></div>
          <div><p className="text-xs text-slate-400">{t.risk}</p><p className={`font-display text-4xl ${RISK[a.injury_risk]}`}>{Math.round(a.injury_probability * 100)}%</p></div>
          <div><p className="text-xs text-slate-400">{t.performance}</p><p className="font-display text-4xl text-lime">{Math.round(a.performance_index)}</p></div>
        </div>
        <p className="mt-4 text-sm font-semibold">{t.why}</p>
        <ul className="mt-1 list-disc pl-5 text-sm text-slate-400">{a.explanations.map((e) => <li key={e}>{e}</li>)}</ul>
      </Card>
      <Card>
        <p className="text-sm font-semibold">{t.recs}</p>
        {a.recommendations.map((r) => <p key={r.title} className="mt-2 text-sm"><b>{r.title}</b> <span className="text-slate-400">— {r.detail}</span></p>)}
      </Card>
      <Card className="md:col-span-2">
        <p className="mb-3 text-sm font-semibold">{t.plan}</p>
        <div className="grid gap-3 md:grid-cols-3">
          {a.plan_72h.map((d) => (
            <div key={d.day} className="rounded-xl border border-line p-3">
              <p className="text-xs uppercase text-slate-400">Day {d.day} · {d.intensity}</p>
              <p className="font-semibold">{d.focus}</p><p className="text-sm text-slate-400">{d.detail}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function AthleteView({ id, t }) {
  const [d, setD] = useState(null);
  useEffect(() => { api.athlete(id).then(setD).catch(console.error); }, [id]);
  if (!d) return <p className="text-slate-400">Loading…</p>;
  const data = d.trend.map((p) => ({ ...p, injury: p.injury_probability != null ? Math.round(p.injury_probability * 100) : null }));
  return (
    <div className="space-y-4">
      <h2 className="font-display text-2xl">{d.athlete.name} <span className="text-base text-slate-400">· {d.athlete.sport}</span></h2>
      <Analysis a={d.latest} t={t} />
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <p className="mb-2 text-sm font-semibold">{t.readiness} / {t.performance}</p>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={data}><CartesianGrid stroke="#1c3344" /><XAxis dataKey="date" stroke="#8aa1b1" fontSize={10} /><YAxis domain={[0, 100]} stroke="#8aa1b1" fontSize={10} />
              <Tooltip contentStyle={{ background: "#0c1a24", border: "1px solid #1c3344" }} /><Legend />
              <Line dataKey="readiness_score" name={t.readiness} stroke="#7dffc3" dot={false} strokeWidth={2} />
              <Line dataKey="performance_index" name={t.performance} stroke="#c8f542" dot={false} strokeWidth={2} />
              <Line dataKey="injury" name={`${t.risk} %`} stroke="#ff6b6b" strokeDasharray="5 5" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
        <Card>
          <p className="mb-2 text-sm font-semibold">{t.load} (sRPE) · ACWR</p>
          <ResponsiveContainer width="100%" height={240}>
            <ComposedChart data={data}><CartesianGrid stroke="#1c3344" /><XAxis dataKey="date" stroke="#8aa1b1" fontSize={10} />
              <YAxis yAxisId="l" stroke="#8aa1b1" fontSize={10} /><YAxis yAxisId="r" orientation="right" domain={[0, 2.5]} stroke="#8aa1b1" fontSize={10} />
              <ReferenceArea yAxisId="r" y1={0.8} y2={1.3} fill="#7dffc3" fillOpacity={0.08} />
              <Tooltip contentStyle={{ background: "#0c1a24", border: "1px solid #1c3344" }} /><Legend />
              <Bar yAxisId="l" dataKey="srpe_load" name="sRPE" fill="#6cc4ff" fillOpacity={0.4} />
              <Line yAxisId="l" dataKey="acute_7d" name="Acute 7d" stroke="#ffb25a" dot={false} />
              <Line yAxisId="r" dataKey="acwr" name="ACWR" stroke="#c8f542" dot={false} strokeWidth={2} />
            </ComposedChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}

function Lab({ athletes, t }) {
  const [form, setForm] = useState({ sport: "Athletics", sleep_hours: 7.2, rpe: 7, duration_min: 75, wellness: 7, sessions_last_7: 5, rest_days_last_7: 2 });
  const [res, setRes] = useState(null);
  useEffect(() => {
    const h = setTimeout(() => api.analyze(form).then(setRes).catch(console.error), 200);
    return () => clearTimeout(h);
  }, [form]);
  return (
    <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
      <Card className="space-y-3">
        <select className="w-full rounded-xl p-2" value={form.athlete_id || ""} onChange={(e) => setForm({ ...form, athlete_id: e.target.value ? Number(e.target.value) : undefined })}>
          <option value="">Guest athlete</option>{athletes.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
        </select>
        {SLIDERS.map(([k, min, max, step]) => (
          <label key={k} className="block text-xs text-slate-400">{k.replaceAll("_", " ")} <b className="float-right text-slate-100">{form[k]}</b>
            <input type="range" className="w-full accent-lime" min={min} max={max} step={step} value={form[k]} onChange={(e) => setForm({ ...form, [k]: Number(e.target.value) })} />
          </label>
        ))}
      </Card>
      <Analysis a={res} t={t} />
    </div>
  );
}

export default function App() {
  const [lang, setLang] = useState("en");
  const [tab, setTab] = useState("dashboard");
  const [stats, setStats] = useState(null);
  const [athletes, setAthletes] = useState([]);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState("");
  const t = useMemo(() => T[lang], [lang]);

  useEffect(() => {
    Promise.all([api.stats(), api.athletes()]).then(([s, a]) => { setStats(s); setAthletes(a); }).catch((e) => setError(e.message));
  }, []);

  return (
    <div className="mx-auto max-w-7xl px-5 py-6">
      <header className="mb-6 flex flex-wrap items-center gap-3">
        <div className="grid h-10 w-10 place-items-center rounded-2xl bg-lime font-display text-ink">AL</div>
        <p className="font-display text-xl">Athlete Lens</p>
        <nav className="ml-auto flex gap-1">
          {["dashboard", "roster", "lab"].map((k) => (
            <button key={k} onClick={() => setTab(k)} className={`rounded-full px-4 py-2 text-sm ${tab === k ? "bg-lime/15 text-lime" : "text-slate-400"}`}>{t[k]}</button>
          ))}
        </nav>
        <button onClick={() => setLang(lang === "en" ? "ta" : "en")} className="rounded-full border border-line px-3 py-1 text-xs">{lang === "en" ? "தமிழ்" : "English"}</button>
      </header>
      {error && <p className="mb-4 rounded-xl bg-red-500/10 p-3 text-red-300">{error}</p>}

      {tab === "dashboard" && stats && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <Kpi label={t.athletes} value={stats.athlete_count} /><Kpi label={t.sessions} value={stats.session_count} />
            <Kpi label={t.readiness} value={stats.avg_readiness} tone="text-mint" /><Kpi label={t.highRisk} value={stats.high_risk} tone="text-red-400" />
          </div>
          <Card>
            <p className="mb-2 font-semibold">{t.watch}</p>
            {stats.watchlist.map((w) => (
              <button key={w.athlete_id} onClick={() => { setSelected(w.athlete_id); setTab("roster"); }} className="flex w-full justify-between border-b border-line py-2 text-left text-sm">
                <span>{w.name} <span className="text-slate-400">· {w.sport}</span></span>
                <span className={RISK[w.injury_risk]}>{w.injury_risk} · {Math.round(w.injury_probability * 100)}%</span>
              </button>
            ))}
          </Card>
        </div>
      )}

      {tab === "roster" && (
        <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
          <Card className="space-y-1">
            {athletes.map((a) => (
              <button key={a.id} onClick={() => setSelected(a.id)} className={`block w-full rounded-xl px-3 py-2 text-left text-sm ${selected === a.id ? "bg-lime/10" : ""}`}>
                {a.name} <span className={`float-right ${RISK[a.latest_risk] || ""}`}>{a.latest_readiness != null ? Math.round(a.latest_readiness) : "—"}</span>
              </button>
            ))}
          </Card>
          {selected ? <AthleteView id={selected} t={t} /> : <p className="text-slate-400">{t.select}</p>}
        </div>
      )}

      {tab === "lab" && <Lab athletes={athletes} t={t} />}
    </div>
  );
}
