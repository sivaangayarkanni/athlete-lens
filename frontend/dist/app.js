/* Athlete Lens — production UI (no build step, no external JS). */
(() => {
  "use strict";

  const $app = document.getElementById("app");
  const SPORTS = ["Athletics", "Kabaddi", "Kho-Kho", "Football", "Hockey", "Volleyball", "Badminton", "Wrestling"];
  const RISK_COLOR = { low: "var(--mint)", moderate: "var(--ember)", high: "var(--red)", none: "#2b4a60" };
  const BAND_COLOR = { green: "var(--mint)", amber: "var(--ember)", red: "var(--red)" };
  const INTENSITY_COLOR = { rest: "var(--sky)", low: "var(--mint)", moderate: "var(--ember)", high: "var(--red)" };
  const PRIORITY_COLOR = { now: "var(--red)", this_week: "var(--ember)", monitor: "var(--sky)" };

  // ------------------------------------------------------------------ utils
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
  const pct = (v) => (v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`);
  const initials = (n) => esc(n.split(/\s+/).map((p) => p[0]).slice(0, 2).join("").toUpperCase());
  const today = () => new Date(Date.now() - new Date().getTimezoneOffset() * 60000).toISOString().slice(0, 10);
  const shortDate = (iso) => new Date(iso + "T00:00:00").toLocaleDateString(undefined, { day: "numeric", month: "short" });
  const badge = (risk) => `<span class="badge ${esc(risk || "none")}">${esc(risk || "no data")}</span>`;
  const readinessColor = (r) => (r == null ? "#2b4a60" : r >= 65 ? "var(--mint)" : r >= 45 ? "var(--ember)" : "var(--red)");
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

  async function api(path, opts = {}) {
    const init = { ...opts, headers: { ...(opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {}), ...(opts.headers || {}) } };
    const res = await fetch(path, init);
    if (res.status === 204) return null;
    let body = null;
    try { body = await res.json(); } catch { /* non-JSON */ }
    if (!res.ok) throw new Error((body && body.detail) || `${res.status} ${res.statusText}`);
    return body;
  }

  let toastTimer;
  function toast(msg, isError = false) {
    const el = document.getElementById("toast");
    el.textContent = msg;
    el.className = `toast show${isError ? " error" : ""}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (el.className = "toast"), 3200);
  }

  const ring = (value, color, size = "", label = "") =>
    `<div class="ring ${size}" style="--v:${Math.max(0, Math.min(100, value ?? 0))};--c:${color}"><span>${fmt(value)}${label ? `<small>${esc(label)}</small>` : ""}</span></div>`;
  const metricBar = (name, v) => `<div class="metric"><span class="muted">${esc(name)}</span><div class="bar"><i style="width:${Math.max(0, Math.min(100, v))}%"></i></div><span>${fmt(v)}</span></div>`;
  const errorBox = (e) => `<div class="error-box"><b>Something went wrong.</b> ${esc(e.message || e)}</div>`;

  // ------------------------------------------------------------------ charts (SVG)
  function lineChart({ labels, series, yMin, yMax, band, bars, height = 220, yFmt = (v) => fmt(v) }) {
    if (!labels.length) return `<div class="chart-empty">No sessions yet.</div>`;
    const W = 640, H = height, P = { l: 40, r: 12, t: 12, b: 26 };
    const all = series.flatMap((s) => s.values).concat(bars ? bars.values : []).filter((v) => v != null);
    let lo = yMin ?? Math.min(...all), hi = yMax ?? Math.max(...all);
    if (band) { lo = Math.min(lo, band.from); hi = Math.max(hi, band.to); }
    if (hi === lo) { hi += 1; lo -= 1; }
    const pad = (hi - lo) * 0.08; if (yMin === undefined) lo -= pad; if (yMax === undefined) hi += pad;
    const n = labels.length;
    const x = (i) => P.l + (n === 1 ? (W - P.l - P.r) / 2 : (i * (W - P.l - P.r)) / (n - 1));
    const y = (v) => P.t + (1 - (v - lo) / (hi - lo)) * (H - P.t - P.b);
    const ticks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
    let svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img"><g class="grid">`;
    ticks.forEach((t) => { svg += `<line x1="${P.l}" x2="${W - P.r}" y1="${y(t)}" y2="${y(t)}"/><text x="${P.l - 6}" y="${y(t) + 3}" text-anchor="end">${esc(yFmt(t))}</text>`; });
    svg += `</g>`;
    if (band) svg += `<rect x="${P.l}" width="${W - P.l - P.r}" y="${y(band.to)}" height="${y(band.from) - y(band.to)}" fill="${band.color}" opacity=".10"/><text x="${W - P.r - 4}" y="${y(band.to) + 11}" text-anchor="end" style="fill:${band.color}">${esc(band.label || "")}</text>`;
    if (bars) {
      const bw = Math.max(4, Math.min(22, ((W - P.l - P.r) / n) * 0.55));
      bars.values.forEach((v, i) => { if (v != null) svg += `<rect x="${x(i) - bw / 2}" y="${y(v)}" width="${bw}" height="${y(Math.max(lo, 0)) - y(v)}" rx="3" fill="${bars.color}" opacity=".35"><title>${esc(labels[i])}: ${esc(bars.name)} ${fmt(v)}</title></rect>`; });
    }
    series.forEach((s) => {
      const pts = s.values.map((v, i) => (v == null ? null : [x(i), y(v)])).filter(Boolean);
      if (!pts.length) return;
      svg += `<polyline fill="none" stroke="${s.color}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round" ${s.dash ? 'stroke-dasharray="5 5"' : ""} points="${pts.map((p) => p.join(",")).join(" ")}"/>`;
      s.values.forEach((v, i) => { if (v != null) svg += `<circle class="dot" cx="${x(i)}" cy="${y(v)}" r="3.2" fill="${s.color}"><title>${esc(labels[i])}: ${esc(s.name)} ${esc(yFmt(v))}</title></circle>`; });
    });
    const step = Math.ceil(n / 8);
    labels.forEach((l, i) => { if (i % step === 0 || i === n - 1) svg += `<text x="${x(i)}" y="${H - 8}" text-anchor="middle">${esc(l)}</text>`; });
    svg += `</svg>`;
    const legend = series.map((s) => `<span style="--c:${s.color}">${esc(s.name)}</span>`).concat(bars ? [`<span style="--c:${bars.color}">${esc(bars.name)}</span>`] : []).join("");
    return svg + `<div class="legend" style="margin-top:8px">${legend}</div>`;
  }

  function hbars(items, color = "var(--lime)") {
    const max = Math.max(...items.map((i) => i.value), 0.0001);
    return items.map((i) => `<div class="metric" style="grid-template-columns:150px 1fr 54px"><span class="muted">${esc(i.label)}</span><div class="bar"><i style="width:${(i.value / max) * 100}%;background:${color}"></i></div><span>${esc(i.text ?? fmt(i.value, 3))}</span></div>`).join("");
  }

  // ------------------------------------------------------------------ shared analysis panel
  function analysisPanel(a) {
    const plan = a.plan_72h.map((d) => `
      <div class="plan-day" style="--c:${INTENSITY_COLOR[d.intensity]}">
        <div class="d">Day ${d.day} · ${esc(d.intensity)}${d.minutes ? ` · ${d.minutes} min` : ""}</div>
        <h3>${esc(d.focus)}</h3><p class="muted small">${esc(d.detail)}</p>
      </div>`).join("");
    const recs = a.recommendations.map((r) => `
      <div class="rec" style="--c:${PRIORITY_COLOR[r.priority]}">
        <div class="row"><span class="t">${esc(r.title)}</span><span class="badge tag">${esc(r.priority.replace("_", " "))}</span></div>
        <p class="muted small">${esc(r.detail)}</p>
      </div>`).join("");
    return `
      <div class="grid g3">
        <div class="card"><div class="row" style="gap:18px">${ring(a.readiness_score, BAND_COLOR[a.readiness_band], "lg", "readiness")}
          <div class="stack"><span class="badge ${a.readiness_band}">${esc(a.readiness_band)}</span><span class="muted small">0-100 composite of recovery, load, performance and wellness.</span></div></div></div>
        <div class="card"><div class="row" style="gap:18px">${ring(a.injury_probability * 100, RISK_COLOR[a.injury_risk], "lg", "injury %")}
          <div class="stack">${badge(a.injury_risk)}${a.overtraining ? '<span class="badge high">overtraining flag</span>' : ""}<span class="muted small">RandomForest probability + load rules.</span></div></div></div>
        <div class="card"><div class="row" style="gap:18px">${ring(a.performance_index, "var(--lime)", "lg", "perf index")}
          <div class="stack"><span class="muted small">GradientBoosting performance index from speed, power, HR and load.</span></div></div></div>
      </div>
      <div class="grid g2" style="margin-top:16px">
        <div class="card"><div class="card-head"><h2>Component scores</h2></div><div class="stack">
          ${metricBar("Load", a.load_score)}${metricBar("Recovery", a.recovery_score)}${metricBar("Speed", a.speed_score)}${metricBar("Power", a.power_score)}</div>
          <h3 style="margin:18px 0 8px">Why</h3><ul class="clean">${a.explanations.map((e) => `<li>${esc(e)}</li>`).join("")}</ul></div>
        <div class="card"><div class="card-head"><h2>Recommendations</h2></div><div class="stack">${recs}</div></div>
      </div>
      <div class="card" style="margin-top:16px"><div class="card-head"><h2>Next 72 hours</h2><span class="muted small">rule-based plan from the model outputs</span></div><div class="plan">${plan}</div></div>`;
  }

  // ------------------------------------------------------------------ views
  async function viewDashboard() {
    const s = await api("/api/stats");
    const total = Math.max(1, s.athlete_count);
    const stacked = ["high", "moderate", "low", "none"].map((k) => `<i style="width:${(s.risk_bands[k] / total) * 100}%;background:${RISK_COLOR[k]}" title="${k}: ${s.risk_bands[k]}"></i>`).join("");
    const watch = s.watchlist.map((w) => `
      <tr class="click" data-href="#/athlete/${w.athlete_id}"><td><b>${esc(w.name)}</b><div class="muted small">${esc(w.sport)}</div></td>
      <td>${badge(w.injury_risk)}${w.overtraining ? ' <span class="badge high">OT</span>' : ""}</td><td class="mono">${pct(w.injury_probability)}</td>
      <td class="mono" style="color:${readinessColor(w.readiness_score)}">${fmt(w.readiness_score)}</td></tr>`).join("");
    const recent = s.recent.map((r) => `
      <tr class="click" data-href="#/athlete/${r.athlete_id}"><td>${shortDate(r.date)}</td><td>${esc(r.name)}</td>
      <td class="mono">${fmt(r.duration_min)}′ · RPE ${fmt(r.rpe, 1)}</td><td>${badge(r.injury_risk)}</td></tr>`).join("");
    const sports = hbars(s.sports.map((x) => ({ label: x.sport, value: x.count, text: String(x.count) })), "var(--mint)");
    $app.innerHTML = `
      <div class="page-head"><div><h1>Squad overview</h1><p class="muted">Who is ready, who is quietly overreaching, and what to change this week.</p></div>
        <div class="row"><a class="btn" href="#/lab">Open analysis lab</a><a class="btn primary" href="#/roster?new=1">+ Add athlete</a></div></div>
      <div class="grid g4">
        <div class="card kpi"><div class="label">Athletes</div><div class="value">${s.athlete_count}</div><div class="sub">${s.sports.length} sports</div></div>
        <div class="card kpi"><div class="label">Sessions logged</div><div class="value">${s.session_count}</div><div class="sub">all scored by the models</div></div>
        <div class="card kpi"><div class="label">Avg readiness</div><div class="value" style="color:${readinessColor(s.avg_readiness)}">${fmt(s.avg_readiness)}</div><div class="sub">latest session per athlete</div></div>
        <div class="card kpi"><div class="label">High risk</div><div class="value" style="color:var(--red)">${s.high_risk}</div><div class="sub">${s.moderate_risk} moderate</div></div>
      </div>
      <div class="card" style="margin-top:16px"><div class="card-head"><h2>Injury-risk bands</h2><div class="legend">
        <span style="--c:${RISK_COLOR.high}">High ${s.risk_bands.high}</span><span style="--c:${RISK_COLOR.moderate}">Moderate ${s.risk_bands.moderate}</span>
        <span style="--c:${RISK_COLOR.low}">Low ${s.risk_bands.low}</span><span style="--c:${RISK_COLOR.none}">No data ${s.risk_bands.none}</span></div></div>
        <div class="stacked">${stacked}</div></div>
      <div class="grid g3" style="margin-top:16px">
        <div class="card span2"><div class="card-head"><h2>Watchlist</h2><span class="muted small">sorted by band, then probability</span></div>
          ${watch ? `<div class="table-wrap"><table><thead><tr><th>Athlete</th><th>Band</th><th>Injury p</th><th>Readiness</th></tr></thead><tbody>${watch}</tbody></table></div>` : '<div class="empty">No scored sessions yet.</div>'}</div>
        <div class="card"><div class="card-head"><h2>Sports</h2></div><div class="stack">${sports || '<div class="empty">No athletes.</div>'}</div></div>
      </div>
      <div class="card" style="margin-top:16px"><div class="card-head"><h2>Recent sessions</h2></div>
        ${recent ? `<div class="table-wrap"><table><thead><tr><th>Date</th><th>Athlete</th><th>Session</th><th>Band</th></tr></thead><tbody>${recent}</tbody></table></div>` : '<div class="empty">No sessions yet — log one from an athlete page or import a CSV.</div>'}</div>`;
  }

  function athleteForm() {
    return `<form id="athlete-form" class="stack">
      <div class="form-grid">
        <label class="field">Name<input name="name" required minlength="2" maxlength="120" placeholder="e.g. Priya N"></label>
        <label class="field">Sport<select name="sport">${SPORTS.map((s) => `<option>${s}</option>`).join("")}</select></label>
        <label class="field">Role / event<input name="role" maxlength="60" value="Athlete"></label>
        <label class="field">Sex<select name="sex"><option value="F">F</option><option value="M">M</option><option value="Other">Other</option></select></label>
        <label class="field">Age<input name="age" type="number" min="10" max="60" value="18" required></label>
        <label class="field">Height (cm)<input name="height_cm" type="number" min="120" max="230" step="0.1" value="165" required></label>
        <label class="field">Weight (kg)<input name="weight_kg" type="number" min="30" max="160" step="0.1" value="55" required></label>
        <label class="field">Years training<input name="years_training" type="number" min="0" max="30" step="0.5" value="2"></label>
        <label class="field">Previous injuries<input name="previous_injuries" type="number" min="0" max="20" value="0"></label>
        <label class="field">City<input name="city" maxlength="80" value="Coimbatore"></label>
        <label class="field">Academy<input name="academy" maxlength="120" value="District Sports Academy"></label>
      </div>
      <label class="field">Notes<textarea name="notes" rows="2" maxlength="1000"></textarea></label>
      <div class="row"><span class="spacer"></span><button type="button" class="btn ghost" data-close>Cancel</button><button class="btn primary">Save athlete</button></div>
    </form>`;
  }

  const formData = (form, numeric) => {
    const out = {};
    new FormData(form).forEach((v, k) => { if (v !== "") out[k] = numeric.includes(k) ? Number(v) : v; });
    return out;
  };

  async function viewRoster(params) {
    const qs = new URLSearchParams();
    ["q", "sport", "risk"].forEach((k) => params.get(k) && qs.set(k, params.get(k)));
    const list = await api(`/api/athletes${qs.toString() ? "?" + qs : ""}`);
    const cards = list.map((a) => `
      <div class="card athlete-card" data-href="#/athlete/${a.id}">
        <div class="row"><div class="avatar">${initials(a.name)}</div><div class="spacer" style="min-width:0"><h3>${esc(a.name)}</h3>
          <div class="muted small">${esc(a.sport)} · ${esc(a.role)}</div></div>${ring(a.latest_readiness, readinessColor(a.latest_readiness))}</div>
        <div class="row small muted"><span>${esc(a.city)}</span><span>·</span><span>${a.age} y</span><span>·</span><span>${a.session_count} sessions</span></div>
        <div class="row">${badge(a.latest_risk)}<span class="spacer"></span><span class="muted small">${a.last_session ? "last " + shortDate(a.last_session) : "no sessions"}</span></div>
      </div>`).join("");
    $app.innerHTML = `
      <div class="page-head"><div><h1>Roster</h1><p class="muted">${list.length} athlete${list.length === 1 ? "" : "s"} · readiness ring shows the latest scored session.</p></div>
        <button class="btn primary" id="add">+ Add athlete</button></div>
      <form id="filters" class="card row" style="margin-bottom:16px">
        <input name="q" placeholder="Search name…" value="${esc(params.get("q") || "")}" style="flex:2;min-width:160px">
        <select name="sport" style="flex:1;min-width:140px"><option value="">All sports</option>${SPORTS.map((s) => `<option ${params.get("sport") === s ? "selected" : ""}>${s}</option>`).join("")}</select>
        <select name="risk" style="flex:1;min-width:140px"><option value="">Any risk</option>${["high", "moderate", "low"].map((r) => `<option value="${r}" ${params.get("risk") === r ? "selected" : ""}>${r}</option>`).join("")}</select>
      </form>
      ${cards ? `<div class="grid g3">${cards}</div>` : '<div class="empty">No athletes match. Clear filters or add one.</div>'}
      <dialog id="dlg"><h2 style="margin-bottom:14px">New athlete</h2>${athleteForm()}</dialog>`;
    const dlg = document.getElementById("dlg");
    document.getElementById("add").onclick = () => dlg.showModal();
    if (params.get("new")) dlg.showModal();
    dlg.querySelector("[data-close]").onclick = () => dlg.close();
    const filters = document.getElementById("filters");
    const apply = () => {
      const p = new URLSearchParams();
      new FormData(filters).forEach((v, k) => v && p.set(k, v));
      location.hash = `#/roster${p.toString() ? "?" + p : ""}`;
    };
    filters.addEventListener("change", apply);
    filters.q.addEventListener("input", debounce(apply, 400));
    filters.onsubmit = (e) => { e.preventDefault(); apply(); };
    document.getElementById("athlete-form").onsubmit = async (e) => {
      e.preventDefault();
      try {
        const a = await api("/api/athletes", { method: "POST", body: JSON.stringify(formData(e.target, ["age", "height_cm", "weight_kg", "years_training", "previous_injuries"])) });
        dlg.close(); toast(`${a.name} added`); location.hash = `#/athlete/${a.id}`;
      } catch (err) { toast(err.message, true); }
    };
    if (params.get("q")) { filters.q.focus(); filters.q.setSelectionRange(99, 99); }
  }

  const SESSION_FIELDS = [
    ["duration_min", "Duration (min)", 10, 300, 1, 70],
    ["distance_km", "Distance (km)", 0, 80, 0.1, 5],
    ["sprint_100m_s", "100m (s)", 9.5, 25, 0.01, 14],
    ["vertical_jump_cm", "Vertical jump (cm)", 15, 90, 0.5, 40],
    ["resting_hr", "Resting HR", 38, 110, 1, 64],
    ["session_hr_avg", "Session HR avg", 80, 210, 1, 148],
    ["rpe", "RPE (1-10)", 1, 10, 0.5, 6],
    ["sleep_hours", "Sleep (h)", 3, 12, 0.1, 7.5],
    ["wellness", "Wellness (1-10)", 1, 10, 0.5, 7],
    ["sessions_last_7", "Sessions last 7d", 0, 14, 1, 4],
    ["rest_days_last_7", "Rest days last 7d", 0, 7, 1, 2],
  ];

  async function viewAthlete(id) {
    const d = await api(`/api/athletes/${id}`);
    const a = d.athlete, t = d.trend, last = d.sessions[d.sessions.length - 1];
    const labels = t.map((p) => shortDate(p.date));
    const lastAcwr = t.length ? t[t.length - 1].acwr : null;
    const rows = [...d.sessions].reverse().map((s) => {
      const p = t.find((x) => x.session_id === s.id) || {};
      return `<tr><td>${shortDate(s.session_date)}</td><td class="mono">${fmt(s.duration_min)}′</td><td class="mono">${fmt(s.rpe, 1)}</td>
        <td class="mono">${fmt(s.sprint_100m_s, 2)}</td><td class="mono">${fmt(s.vertical_jump_cm, 1)}</td><td class="mono">${fmt(s.sleep_hours, 1)}</td>
        <td class="mono" style="color:${readinessColor(p.readiness_score)}">${fmt(p.readiness_score)}</td><td>${badge(p.injury_risk)}</td>
        <td><button class="btn sm ghost danger" data-del-session="${s.id}" title="Delete session">✕</button></td></tr>`;
    }).join("");
    $app.innerHTML = `
      <div class="page-head"><div class="row" style="gap:16px"><div class="avatar" style="width:60px;height:60px;font-size:20px">${initials(a.name)}</div>
        <div><h1>${esc(a.name)}</h1><p class="muted">${esc(a.sport)} · ${esc(a.role)} · ${esc(a.academy)}, ${esc(a.city)}</p></div></div>
        <div class="row"><a class="btn" href="#/lab?athlete=${a.id}">What-if in lab</a><button class="btn primary" id="log">+ Log session</button><button class="btn danger" id="del">Delete</button></div></div>
      <div class="card" style="margin-bottom:16px"><dl class="kv" style="grid-template-columns:repeat(auto-fill,minmax(110px,auto) minmax(60px,1fr))">
        <dt>Age</dt><dd>${a.age}</dd><dt>Sex</dt><dd>${esc(a.sex)}</dd><dt>Height</dt><dd>${fmt(a.height_cm)} cm</dd><dt>Weight</dt><dd>${fmt(a.weight_kg, 1)} kg</dd>
        <dt>Training</dt><dd>${fmt(a.years_training, 1)} y</dd><dt>Prev. injuries</dt><dd>${a.previous_injuries}</dd><dt>Sessions</dt><dd>${a.session_count}</dd>
        <dt>ACWR</dt><dd style="color:${lastAcwr == null ? "inherit" : lastAcwr > 1.5 ? "var(--red)" : lastAcwr > 1.3 ? "var(--ember)" : "var(--mint)"}">${fmt(lastAcwr, 2)}</dd></dl>
        ${a.notes ? `<p class="muted small" style="margin-top:10px">${esc(a.notes)}</p>` : ""}</div>
      <section id="log-wrap" hidden class="card" style="margin-bottom:16px"><div class="card-head"><h2>Log a session</h2><span class="muted small">pre-filled from the last session</span></div>
        <form id="session-form" class="stack"><div class="form-grid">
          <label class="field">Date<input type="date" name="session_date" value="${today()}" max="${today()}" required></label>
          ${SESSION_FIELDS.map(([k, l, mn, mx, st, def]) => `<label class="field">${l}<input type="number" name="${k}" min="${mn}" max="${mx}" step="${st}" value="${last ? last[k] : def}" required></label>`).join("")}
        </div><label class="field">Notes<input name="notes" maxlength="1000" placeholder="optional"></label>
        <div class="row"><span class="spacer"></span><button type="button" class="btn ghost" id="cancel-log">Cancel</button><button class="btn primary">Score &amp; save</button></div></form></section>
      ${d.latest ? `<h2 style="margin:4px 0 12px">Latest analysis <span class="muted small">(${shortDate(last.session_date)})</span></h2>${analysisPanel(d.latest)}` : '<div class="empty">No sessions yet — log the first one to get readiness, risk and a 72-hour plan.</div>'}
      ${t.length ? `<div class="grid g2" style="margin-top:16px">
        <div class="card"><div class="card-head"><h2>Readiness &amp; performance</h2></div>${lineChart({ labels, yMin: 0, yMax: 100, series: [
          { name: "Readiness", color: "var(--mint)", values: t.map((p) => p.readiness_score) },
          { name: "Performance index", color: "var(--lime)", values: t.map((p) => p.performance_index) },
          { name: "Injury probability ×100", color: "var(--red)", dash: true, values: t.map((p) => (p.injury_probability == null ? null : p.injury_probability * 100)) }] })}</div>
        <div class="card"><div class="card-head"><h2>Training load (sRPE)</h2><span class="muted small">duration × RPE</span></div>${lineChart({ labels, yMin: 0,
          bars: { name: "Session load", color: "var(--sky)", values: t.map((p) => p.srpe_load) },
          series: [{ name: "Acute 7-day", color: "var(--ember)", values: t.map((p) => p.acute_7d) }, { name: "Chronic (28-day weekly avg)", color: "var(--mint)", values: t.map((p) => p.chronic_28d) }] })}</div>
        <div class="card"><div class="card-head"><h2>Acute : chronic ratio</h2><span class="muted small">0.8–1.3 is the usual sweet spot</span></div>${lineChart({ labels, yMin: 0, yFmt: (v) => fmt(v, 1),
          band: { from: 0.8, to: 1.3, color: "var(--mint)", label: "sweet spot" }, series: [{ name: "ACWR", color: "var(--lime)", values: t.map((p) => p.acwr) }] })}</div>
        <div class="card"><div class="card-head"><h2>Field tests</h2><span class="muted small">lower 100m is better</span></div>${lineChart({ labels, yFmt: (v) => fmt(v, 1), series: [
          { name: "100m (s)", color: "var(--sky)", values: t.map((p) => p.sprint_100m_s) },
          { name: "Sleep (h)", color: "var(--mint)", dash: true, values: t.map((p) => p.sleep_hours) }] })}
          ${lineChart({ labels, height: 150, series: [{ name: "Vertical jump (cm)", color: "var(--lime)", values: t.map((p) => p.vertical_jump_cm) }] })}</div>
      </div>` : ""}
      <div class="card" style="margin-top:16px"><div class="card-head"><h2>Session log</h2><span class="muted small">${d.sessions.length} sessions</span></div>
        ${rows ? `<div class="table-wrap"><table><thead><tr><th>Date</th><th>Dur</th><th>RPE</th><th>100m</th><th>Jump</th><th>Sleep</th><th>Readiness</th><th>Band</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>` : '<div class="empty">No sessions.</div>'}</div>`;

    const wrap = document.getElementById("log-wrap");
    document.getElementById("log").onclick = () => { wrap.hidden = false; wrap.scrollIntoView({ behavior: "smooth", block: "start" }); };
    document.getElementById("cancel-log").onclick = () => (wrap.hidden = true);
    document.getElementById("session-form").onsubmit = async (e) => {
      e.preventDefault();
      const body = formData(e.target, SESSION_FIELDS.map((f) => f[0]));
      body.athlete_id = a.id;
      try {
        const r = await api("/api/sessions", { method: "POST", body: JSON.stringify(body) });
        toast(`Session scored: readiness ${fmt(r.analysis.readiness_score)}, ${r.analysis.injury_risk} risk`);
        route();
      } catch (err) { toast(err.message, true); }
    };
    document.getElementById("del").onclick = async () => {
      if (!confirm(`Delete ${a.name} and all sessions? This cannot be undone.`)) return;
      try { await api(`/api/athletes/${a.id}`, { method: "DELETE" }); toast(`${a.name} deleted`); location.hash = "#/roster"; } catch (err) { toast(err.message, true); }
    };
    $app.querySelectorAll("[data-del-session]").forEach((b) => (b.onclick = async (ev) => {
      ev.stopPropagation();
      if (!confirm("Delete this session?")) return;
      try { await api(`/api/sessions/${b.dataset.delSession}`, { method: "DELETE" }); toast("Session deleted"); route(); } catch (err) { toast(err.message, true); }
    }));
  }

  async function viewLab(params) {
    const athletes = await api("/api/athletes");
    const sel = params.get("athlete") || "";
    let base = { age: 18, sex: "F", sport: "Athletics", years_training: 3, height_cm: 165, weight_kg: 55, previous_injuries: 0 };
    SESSION_FIELDS.forEach(([k, , , , , def]) => (base[k] = def));
    if (sel) {
      try {
        const d = await api(`/api/athletes/${sel}`);
        ["age", "sex", "sport", "years_training", "height_cm", "weight_kg", "previous_injuries"].forEach((k) => (base[k] = d.athlete[k]));
        const last = d.sessions[d.sessions.length - 1];
        if (last) SESSION_FIELDS.forEach(([k]) => (base[k] = last[k]));
      } catch (e) { toast(e.message, true); }
    }
    const sliders = SESSION_FIELDS.map(([k, l, mn, mx, st]) => `<label class="slider">${l}<b id="v-${k}">${base[k]}</b><input type="range" name="${k}" min="${mn}" max="${mx}" step="${st}" value="${base[k]}"></label>`).join("");
    $app.innerHTML = `
      <div class="page-head"><div><h1>Analysis lab</h1><p class="muted">Move the sliders to see how sleep, load and rest change readiness and risk. Nothing is saved.</p></div>
        <select id="who" style="max-width:280px"><option value="">Guest athlete (custom profile)</option>${athletes.map((a) => `<option value="${a.id}" ${String(a.id) === sel ? "selected" : ""}>${esc(a.name)} — ${esc(a.sport)}</option>`).join("")}</select></div>
      <div class="grid" style="grid-template-columns:minmax(260px,340px) 1fr;align-items:start" id="lab-grid">
        <form id="lab" class="card stack">
          <h2>Profile</h2>
          <div class="form-grid" style="grid-template-columns:1fr 1fr">
            <label class="field">Sport<select name="sport">${SPORTS.map((s) => `<option ${base.sport === s ? "selected" : ""}>${s}</option>`).join("")}</select></label>
            <label class="field">Sex<select name="sex">${["F", "M", "Other"].map((s) => `<option ${base.sex === s ? "selected" : ""}>${s}</option>`).join("")}</select></label>
            <label class="field">Age<input type="number" name="age" min="10" max="60" value="${base.age}"></label>
            <label class="field">Prev. injuries<input type="number" name="previous_injuries" min="0" max="20" value="${base.previous_injuries}"></label>
            <label class="field">Height (cm)<input type="number" name="height_cm" min="120" max="230" value="${base.height_cm}"></label>
            <label class="field">Weight (kg)<input type="number" name="weight_kg" min="30" max="160" value="${base.weight_kg}"></label>
          </div>
          <input type="hidden" name="years_training" value="${base.years_training}">
          <h2 style="margin-top:6px">Session & week</h2>${sliders}
          <button type="button" class="btn ghost" id="reset">Reset to baseline</button>
        </form>
        <div id="lab-out"><div class="loading">Scoring…</div></div>
      </div>`;
    if (window.matchMedia("(max-width:860px)").matches) document.getElementById("lab-grid").style.gridTemplateColumns = "1fr";
    const form = document.getElementById("lab"), out = document.getElementById("lab-out");
    const numeric = ["age", "years_training", "height_cm", "weight_kg", "previous_injuries", ...SESSION_FIELDS.map((f) => f[0])];
    const run = debounce(async () => {
      const body = formData(form, numeric);
      SESSION_FIELDS.forEach(([k]) => (document.getElementById(`v-${k}`).textContent = body[k]));
      try { out.innerHTML = analysisPanel(await api("/api/analyze", { method: "POST", body: JSON.stringify(body) })); } catch (e) { out.innerHTML = errorBox(e); }
    }, 180);
    form.addEventListener("input", run);
    document.getElementById("who").onchange = (e) => (location.hash = e.target.value ? `#/lab?athlete=${e.target.value}` : "#/lab");
    document.getElementById("reset").onclick = () => route();
    run();
  }

  const SAMPLE_CSV = `name,sport,sex,age,height_cm,weight_kg,years_training,previous_injuries,city,session_date,duration_min,distance_km,sprint_100m_s,vertical_jump_cm,resting_hr,session_hr_avg,rpe,sleep_hours,wellness,sessions_last_7,rest_days_last_7
Priya N,Athletics,F,18,162,51,4,0,Theni,2026-09-01,70,6.2,13.9,41,64,146,6,7.5,8,4,2
Arun Kumar,Football,M,20,176,68,5,1,Dindigul,2026-09-02,88,8.1,12.8,49,60,158,8,6.0,5,7,1
Lakshmi Devi,Kabaddi,F,19,159,54,3,0,Pudukkottai,2026-09-03,65,2.1,14.6,37,70,149,7,7.0,7,5,2
`;

  async function viewImport() {
    $app.innerHTML = `
      <div class="page-head"><div><h1>CSV import</h1><p class="muted">Bulk-load a hostel register or Excel export. One row = one session; athletes are matched by name + sport.</p></div>
        <a class="btn" id="sample" download="athlete_lens_sample.csv">Download sample CSV</a></div>
      <div class="grid g2">
        <div class="card"><label class="drop" id="drop"><input type="file" id="file" accept=".csv,text/csv" hidden>
          <h2>Drop a .csv here</h2><p class="muted" style="margin-top:6px">or click to choose a file (max 2 MB, 2,000 rows)</p></label>
          <div id="result" style="margin-top:16px"></div></div>
        <div class="card"><h2 style="margin-bottom:10px">Columns</h2>
          <p class="small"><b>Required:</b> <span class="muted">name, sport, age, height_cm, weight_kg, session_date (YYYY-MM-DD), duration_min</span></p>
          <p class="small" style="margin-top:8px"><b>Optional:</b> <span class="muted">sex, role, city, academy, years_training, previous_injuries, distance_km, sprint_100m_s, vertical_jump_cm, resting_hr, session_hr_avg, rpe, sleep_hours, wellness, sessions_last_7, rest_days_last_7, notes</span></p>
          <p class="small" style="margin-top:8px"><b>Sports:</b> <span class="muted">${SPORTS.join(", ")}</span></p>
          <p class="small muted" style="margin-top:8px">Invalid rows are skipped and reported with their line number; valid rows are still imported and scored.</p></div>
      </div>`;
    document.getElementById("sample").href = URL.createObjectURL(new Blob([SAMPLE_CSV], { type: "text/csv" }));
    const drop = document.getElementById("drop"), input = document.getElementById("file"), result = document.getElementById("result");
    const upload = async (file) => {
      if (!file) return;
      result.innerHTML = `<div class="loading">Uploading ${esc(file.name)}…</div>`;
      const fd = new FormData(); fd.append("file", file);
      try {
        const r = await api("/api/upload-csv", { method: "POST", body: fd });
        result.innerHTML = `<div class="grid g3"><div class="kpi"><div class="label">Rows</div><div class="value">${r.rows}</div></div>
          <div class="kpi"><div class="label">New athletes</div><div class="value">${r.athletes_created}</div></div>
          <div class="kpi"><div class="label">Sessions</div><div class="value" style="color:var(--mint)">${r.sessions_created}</div></div></div>
          ${r.errors.length ? `<div class="error-box" style="margin-top:12px"><b>${r.errors.length} row(s) skipped</b><ul class="clean">${r.errors.map((e) => `<li>Line ${e.line}: ${esc(e.error)}</li>`).join("")}</ul></div>` : ""}
          <a class="btn primary" style="margin-top:12px" href="#/roster">View roster</a>`;
        toast(`Imported ${r.sessions_created} session(s)`);
      } catch (e) { result.innerHTML = errorBox(e); }
      input.value = "";
    };
    input.onchange = () => upload(input.files[0]);
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));
  }

  async function viewModel() {
    const m = await api("/api/model");
    const inj = m.metrics.injury || {}, perf = m.metrics.performance || {};
    const row = (k, v, note = "") => `<tr><td>${esc(k)}</td><td class="mono"><b>${esc(v ?? "—")}</b></td><td class="muted small">${esc(note)}</td></tr>`;
    $app.innerHTML = `
      <div class="page-head"><div><h1>Model card</h1><p class="muted">Honest numbers, computed on a held-out split when the models were trained (${esc(m.trained_at || "")}, scikit-learn ${esc(m.sklearn_version)}).</p></div></div>
      <div class="grid g2">
        <div class="card"><div class="card-head"><h2>Injury risk</h2><span class="badge tag">classifier</span></div><p class="muted small" style="margin-bottom:10px">${esc(m.injury_model)}</p>
          <div class="table-wrap"><table><tbody>
            ${row("Held-out ROC AUC", inj.holdout_roc_auc, "0.5 = chance")}
            ${row("Held-out accuracy", inj.holdout_accuracy, `majority-class baseline ${inj.baseline_majority_accuracy}`)}
            ${row("Balanced accuracy", inj.holdout_balanced_accuracy)}
            ${row("Recall (injured)", inj.holdout_recall, "share of positive cases flagged at p≥0.5")}
            ${row("Brier score", inj.holdout_brier, "lower is better")}
            ${row("Train accuracy", inj.train_accuracy, "gap vs held-out = overfit")}
            ${row("Positive rate", inj.positive_rate, "share of injury labels in cohort")}
          </tbody></table></div></div>
        <div class="card"><div class="card-head"><h2>Performance index</h2><span class="badge tag">regressor</span></div><p class="muted small" style="margin-bottom:10px">${esc(m.performance_model)}</p>
          <div class="table-wrap"><table><tbody>
            ${row("Held-out MAE", perf.holdout_mae, `predict-the-mean baseline ${perf.baseline_mean_mae}`)}
            ${row("Held-out R²", perf.holdout_r2)}
            ${row("Train MAE", perf.train_mae)}
          </tbody></table></div>
          <h3 style="margin:16px 0 8px">Training data</h3><dl class="kv"><dt>Kind</dt><dd>${esc(m.training_data.kind)}</dd><dt>Cohort</dt><dd>${esc(m.training_data.cohort_size)} athletes</dd>
          <dt>Split</dt><dd>${esc(m.training_data.train_rows)} train / ${esc(m.training_data.test_rows)} test</dd><dt>Seed</dt><dd>${esc(m.training_data.seed)}</dd></dl>
          <p class="muted small" style="margin-top:8px">${esc(m.training_data.description)}</p></div>
      </div>
      <div class="grid g2" style="margin-top:16px">
        <div class="card"><div class="card-head"><h2>What drives injury risk</h2><span class="muted small">RandomForest impurity importance</span></div>
          <div class="stack">${hbars(m.feature_importance.map((f) => ({ label: f.feature, value: f.weight })))}</div></div>
        <div class="card"><h2 style="margin-bottom:10px">Bands</h2><dl class="kv">${Object.entries(m.risk_bands).map(([k, v]) => `<dt>${badge(k)}</dt><dd class="small">${esc(v)}</dd>`).join("")}
          ${Object.entries(m.readiness_bands).map(([k, v]) => `<dt><span class="badge ${k}">readiness ${k}</span></dt><dd class="small">${esc(v)}</dd>`).join("")}</dl>
          <h3 style="margin:16px 0 6px">Intended use</h3><p class="muted small">${esc(m.intended_use)}</p>
          <h3 style="margin:16px 0 6px">Limitations</h3><ul class="clean small">${m.limitations.map((l) => `<li>${esc(l)}</li>`).join("")}</ul></div>
      </div>`;
  }

  // ------------------------------------------------------------------ router
  async function route() {
    const [path, query] = (location.hash.slice(1) || "/").split("?");
    const params = new URLSearchParams(query || "");
    const parts = path.split("/").filter(Boolean);
    const name = parts[0] || "dashboard";
    document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === (name === "athlete" ? "roster" : name)));
    $app.innerHTML = `<div class="loading">Loading…</div>`;
    try {
      if (name === "dashboard") await viewDashboard();
      else if (name === "roster") await viewRoster(params);
      else if (name === "athlete" && parts[1]) await viewAthlete(encodeURIComponent(parts[1]));
      else if (name === "lab") await viewLab(params);
      else if (name === "import") await viewImport();
      else if (name === "model") await viewModel();
      else $app.innerHTML = `<div class="empty">Page not found. <a href="#/">Go to dashboard</a></div>`;
      document.title = `Athlete Lens — ${document.querySelector("#app h1")?.textContent || "Performance Intelligence"}`;
    } catch (e) {
      $app.innerHTML = errorBox(e) + `<p style="margin-top:12px"><a href="#/">Back to dashboard</a></p>`;
    }
  }

  $app.addEventListener("click", (e) => {
    const el = e.target.closest("[data-href]");
    if (el && !e.target.closest("button,a,input,select")) location.hash = el.dataset.href;
  });
  window.addEventListener("hashchange", () => { route(); window.scrollTo(0, 0); });

  (async function boot() {
    // Support clean paths like /athlete/3 by mapping them onto the hash router.
    if (location.pathname !== "/" && !location.hash) { history.replaceState(null, "", "/#" + location.pathname); }
    route();
    const st = document.getElementById("status");
    try {
      const h = await api("/api/health");
      st.className = "status ok";
      st.lastElementChild.textContent = `API ${h.version} · ${h.platform}`;
    } catch { st.className = "status err"; st.lastElementChild.textContent = "API offline"; }
  })();
})();
