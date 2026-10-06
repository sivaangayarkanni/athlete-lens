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

  const ring = (value, color, size = "", label = "", text = null) =>
    `<div class="ring ${size}" style="--v:${Math.max(0, Math.min(100, value ?? 0))};--c:${color}"><span>${text ?? fmt(value)}${label ? `<small>${esc(label)}</small>` : ""}</span></div>`;
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

  // ------------------------------------------------------------------ injury visuals
  const REGIONS = ["hamstring", "quadriceps", "groin", "calf_achilles", "knee", "ankle", "foot", "hip", "lower_back", "shoulder", "elbow_wrist", "head_neck"];
  const REGION_SHORT = { hamstring: "Hamstring", quadriceps: "Quads", groin: "Groin", calf_achilles: "Calf/Achilles", knee: "Knee", ankle: "Ankle", foot: "Foot", hip: "Hip", lower_back: "Low back", shoulder: "Shoulder", elbow_wrist: "Elbow/wrist", head_neck: "Head/neck" };
  const TYPE_SHORT = { muscle_strain: "Muscle strain", ligament_sprain: "Ligament sprain", tendinopathy: "Tendinopathy", overuse_stress: "Overuse / stress", contusion: "Contusion", concussion: "Concussion" };
  const LEVEL_COLOR = { low: "var(--sky)", typical: "var(--mint)", elevated: "var(--ember)", high: "var(--red)" };
  const levelBadge = (l) => `<span class="badge lvl-${esc(l)}">${esc(l)}</span>`;
  // Relative risk (vs sport average) -> colour: blue (below) -> green (typical) -> amber -> red (>= 2.5x)
  function rrColor(rr) {
    if (rr == null) return "#24394a";
    const t = Math.max(0, Math.min(1, (Math.log2(Math.max(rr, 0.01)) + 1) / (Math.log2(2.5) + 1)));
    return `hsl(${Math.round(200 - t * 200)} 85% ${Math.round(58 - t * 6)}%)`;
  }

  // Front/back body map. Each region is one or more shapes tagged with data-region.
  const BODY = {
    front: {
      head_neck: '<circle cx="100" cy="36" r="21"/><rect x="91" y="56" width="18" height="14" rx="5"/>',
      shoulder: '<circle cx="63" cy="82" r="13"/><circle cx="137" cy="82" r="13"/>',
      elbow_wrist: '<ellipse cx="45" cy="166" rx="9" ry="11"/><ellipse cx="155" cy="166" rx="9" ry="11"/><ellipse cx="37" cy="238" rx="8" ry="9"/><ellipse cx="163" cy="238" rx="8" ry="9"/>',
      hip: '<ellipse cx="76" cy="196" rx="11" ry="10"/><ellipse cx="124" cy="196" rx="11" ry="10"/>',
      groin: '<ellipse cx="92" cy="230" rx="6" ry="17"/><ellipse cx="108" cy="230" rx="6" ry="17"/>',
      quadriceps: '<ellipse cx="80" cy="262" rx="11" ry="33"/><ellipse cx="120" cy="262" rx="11" ry="33"/>',
      knee: '<circle cx="83" cy="318" r="11"/><circle cx="117" cy="318" r="11"/>',
      ankle: '<circle cx="84" cy="396" r="7"/><circle cx="116" cy="396" r="7"/>',
      foot: '<ellipse cx="81" cy="414" rx="11" ry="6"/><ellipse cx="119" cy="414" rx="11" ry="6"/>',
    },
    back: {
      head_neck: '<circle cx="100" cy="36" r="21"/><rect x="91" y="56" width="18" height="14" rx="5"/>',
      shoulder: '<ellipse cx="72" cy="92" rx="16" ry="14"/><ellipse cx="128" cy="92" rx="16" ry="14"/>',
      elbow_wrist: '<ellipse cx="45" cy="166" rx="9" ry="11"/><ellipse cx="155" cy="166" rx="9" ry="11"/><ellipse cx="37" cy="238" rx="8" ry="9"/><ellipse cx="163" cy="238" rx="8" ry="9"/>',
      lower_back: '<rect x="80" y="152" width="40" height="34" rx="10"/>',
      hip: '<ellipse cx="84" cy="208" rx="15" ry="13"/><ellipse cx="116" cy="208" rx="15" ry="13"/>',
      hamstring: '<ellipse cx="82" cy="266" rx="12" ry="34"/><ellipse cx="118" cy="266" rx="12" ry="34"/>',
      knee: '<circle cx="83" cy="318" r="9"/><circle cx="117" cy="318" r="9"/>',
      calf_achilles: '<ellipse cx="84" cy="356" rx="10" ry="24"/><ellipse cx="116" cy="356" rx="10" ry="24"/><rect x="81" y="378" width="6" height="12" rx="3"/><rect x="113" y="378" width="6" height="12" rx="3"/>',
      ankle: '<circle cx="84" cy="396" r="6"/><circle cx="116" cy="396" r="6"/>',
      foot: '<ellipse cx="84" cy="413" rx="9" ry="6"/><ellipse cx="116" cy="413" rx="9" ry="6"/>',
    },
  };
  const SILHOUETTE = '<circle cx="100" cy="36" r="23"/><rect x="89" y="54" width="22" height="20" rx="6"/>' +
    '<path d="M58 74 Q100 64 142 74 L150 92 L140 190 Q100 204 60 190 L50 92 Z"/>' +
    '<path d="M50 80 Q40 84 38 100 L34 170 L28 240 L44 242 L52 172 L60 104 Z"/><path d="M150 80 Q160 84 162 100 L166 170 L172 240 L156 242 L148 172 L140 104 Z"/>' +
    '<path d="M62 186 L138 186 L134 230 L124 330 L122 400 L108 400 L104 330 L100 236 L96 330 L92 400 L78 400 L76 330 L66 230 Z"/>' +
    '<ellipse cx="82" cy="414" rx="13" ry="7"/><ellipse cx="118" cy="414" rx="13" ry="7"/>';

  function bodyMap(regions, selected) {
    const by = Object.fromEntries((regions || []).map((r) => [r.region, r]));
    const view = (side) => `<svg class="bodymap" viewBox="0 0 200 430" role="img" aria-label="${side} view">
      <text x="100" y="428" text-anchor="middle" class="bm-label">${side === "front" ? "FRONT" : "BACK"}</text>
      <g class="silhouette">${SILHOUETTE}</g>
      ${Object.entries(BODY[side]).map(([reg, shapes]) => {
        const r = by[reg];
        return `<g class="region${selected === reg ? " sel" : ""}" data-region="${reg}" tabindex="0" style="--fill:${rrColor(r && r.relative_risk)}">
          <title>${esc(r ? `${r.label}: ${(r.probability * 100).toFixed(2)}% (7-day), ${r.relative_risk}x sport average, ${r.level}` : reg)}</title>${shapes}</g>`;
      }).join("")}</svg>`;
    const scale = [0.5, 0.8, 1, 1.3, 2, 2.5].map((v) => `<span><i style="background:${rrColor(v)}"></i>${v}x</span>`).join("");
    return `<div class="bodymap-wrap">${view("front")}${view("back")}</div><div class="rr-scale muted small">risk vs sport average ${scale}</div>`;
  }

  function divergingBars(drivers, max = 8) {
    const items = drivers.slice(0, max);
    const m = Math.max(...items.map((d) => Math.abs(d.contribution)), 1e-6);
    return `<div class="div-bars">${items.map((d) => {
      const w = (Math.abs(d.contribution) / m) * 50;
      const up = d.contribution > 0;
      return `<div class="db-row"><span class="db-label">${esc(d.label)}</span><div class="db-track"><i style="${up ? "left:50%" : `left:${50 - w}%`};width:${w}%;background:${up ? "var(--red)" : "var(--mint)"}"></i></div>
        <span class="db-val mono" style="color:${up ? "var(--red)" : "var(--mint)"}">${up ? "+" : "−"}${(Math.abs(d.contribution) * 100).toFixed(2)} pp</span></div>`;
    }).join("")}</div>`;
  }

  function regionTable(regions, selected) {
    const rows = [...regions].sort((a, b) => b.probability - a.probability).map((r) => `
      <tr class="click${selected === r.region ? " sel" : ""}" data-region="${r.region}"><td><i class="swatch" style="background:${rrColor(r.relative_risk)}"></i>${esc(r.label)}</td>
      <td class="mono">${(r.probability * 100).toFixed(2)}%</td><td class="mono">${fmt(r.relative_risk, 2)}×</td><td>${levelBadge(r.level)}</td>
      <td class="small muted">${esc(r.likely_type_label || "")}</td></tr>`).join("");
    return `<div class="table-wrap"><table><thead><tr><th>Region</th><th>7-day p</th><th>vs sport</th><th>Level</th><th>Likely type</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  function injuryOverview(inj, selected) {
    const o = inj.overall;
    const top = inj.top_regions.map((r) => inj.regions.find((x) => x.region === r));
    const mix = hbars(inj.type_mix.map((t) => ({ label: t.label, value: t.share, text: pct(t.share) })), "var(--sky)");
    const load = inj.load || {};
    return `
      <div class="grid injury-grid">
        <div class="card"><div class="card-head"><h2>Body map</h2><span class="muted small">click a region</span></div>${bodyMap(inj.regions, selected)}</div>
        <div class="stack" style="gap:16px">
          <div class="card"><div class="row" style="gap:18px">${ring(o.probability * 400, RISK_COLOR[o.band], "lg", "7-day risk", (o.probability * 100).toFixed(1) + "%")}
            <div class="stack" style="gap:6px"><div class="muted small">Chance of any time-loss injury in the next ${o.horizon_days} days</div>
            <div class="row">${badge(o.band)}<span class="muted small">${fmt(o.relative_risk, 2)}× the ${esc(inj.sport || "sport")} average (${(o.sport_average * 100).toFixed(1)}%)</span></div>
            <div class="small">Most likely: ${top.map((r) => `<a href="#" data-region="${r.region}"><b>${esc(REGION_SHORT[r.region])}</b></a> <span class="muted">(${(r.probability * 100).toFixed(1)}%, ${esc((r.likely_type_label || "").toLowerCase())})</span>`).join(" · ")}</div>
            ${(inj.elevated_regions || []).length ? `<div class="small">Most elevated vs sport: ${inj.elevated_regions.map((k) => { const r = inj.regions.find((x) => x.region === k); return `<a href="#" data-region="${k}"><b>${esc(REGION_SHORT[k])}</b></a> <span class="muted">(${fmt(r.relative_risk, 1)}×)</span>`; }).join(" · ")}</div>` : ""}
            ${load.history_padded ? `<div class="muted small">⚠ ${load.synthetic_history ? "Synthetic 4-week history built from the form." : "Less than 4 weeks logged: first week repeated to estimate chronic load."}</div>` : ""}</div></div></div>
          <div class="card"><div class="card-head"><h2>All regions</h2><span class="muted small">calibrated 7-day probability, most likely first</span></div>${regionTable(inj.regions, selected)}</div>
        </div>
      </div>
      <div class="grid g3" style="margin-top:16px">
        <div class="card span2"><div class="card-head"><h2>What drives the overall risk</h2><span class="muted small">Shapley contributions vs a typical ${esc(inj.sport || "")} athlete</span></div>
          ${inj.drivers && inj.drivers.any ? `${divergingBars(inj.drivers.any.drivers)}<p class="muted small" style="margin-top:8px">Reference athlete ${pct(inj.drivers.any.reference_probability)} → this athlete ${pct(inj.drivers.any.probability)}. Red pushes risk up, green pulls it down.</p>` : '<div class="empty">No explanation.</div>'}</div>
        <div class="card"><div class="card-head"><h2>Expected injury type</h2></div><div class="stack">${mix}</div>
          <h3 style="margin:16px 0 8px">Load snapshot</h3><dl class="kv">
          <dt>ACWR (EWMA)</dt><dd style="color:${acwrColor(load.acwr)}">${fmt(load.acwr, 2)}</dd><dt>Monotony</dt><dd style="color:${load.monotony > 2 ? "var(--ember)" : "inherit"}">${fmt(load.monotony, 2)}</dd>
          <dt>Strain (7d)</dt><dd>${fmt(load.strain)}</dd><dt>Week-on-week</dt><dd>${load.load_change_wow == null ? "—" : (load.load_change_wow > 0 ? "+" : "") + fmt(load.load_change_wow * 100) + "%"}</dd>
          <dt>Sleep debt 7d</dt><dd>${fmt(load.sleep_debt_7, 1)} h</dd><dt>Matches 7d</dt><dd>${fmt(load.matches_7)}</dd></dl></div>
      </div>
      ${inj.real_data ? `<div class="card real-card" style="margin-top:16px"><div class="card-head"><h2>Real-data cross-check</h2><span class="badge tag">trained on real logs</span></div>
        <div class="row" style="gap:24px"><div><div class="big mono">${fmt(inj.real_data.relative_risk, 2)}×</div><div class="muted small">daily injury rate vs the runner cohort average (${(inj.real_data.cohort_daily_rate * 100).toFixed(2)}%/day)</div></div>
        <p class="muted small" style="flex:1;min-width:240px">${esc(inj.real_data.note)}</p></div></div>` : ""}`;
  }
  const acwrColor = (v) => (v == null ? "inherit" : v > 1.5 ? "var(--red)" : v > 1.3 ? "var(--ember)" : v < 0.8 ? "var(--sky)" : "var(--mint)");

  function regionDetail(d) {
    const g = d.guidance;
    const cfs = d.counterfactuals.map((c) => {
      const ch = c.changes.find((x) => x.target === d.region), any = c.changes.find((x) => x.target === "any");
      const good = ch.relative_change < -0.005;
      return `<tr><td>${esc(c.label)}</td><td class="mono">${(ch.before * 100).toFixed(2)}% → <b style="color:${good ? "var(--mint)" : ch.relative_change > 0.005 ? "var(--red)" : "inherit"}">${(ch.after * 100).toFixed(2)}%</b></td>
        <td class="mono" style="color:${good ? "var(--mint)" : "var(--muted)"}">${ch.relative_change > 0 ? "+" : ""}${fmt(ch.relative_change * 100)}%</td><td class="mono muted">${(any.after * 100).toFixed(1)}%</td></tr>`;
    }).join("");
    const types = Object.entries(d.type_probs || {}).map(([t, v]) => ({ label: TYPE_SHORT[t] || t, value: v, text: pct(v) }));
    const hist = d.history.map((h) => `<li>${esc(TYPE_SHORT[h.injury_type] || h.injury_type)} · ${shortDate(h.onset_date)}${h.return_date ? " → " + shortDate(h.return_date) : " (ongoing)"}</li>`).join("");
    return `<div class="card region-detail" id="region-detail" style="margin-top:16px">
      <div class="card-head"><div><h2>${esc(d.label)}</h2><p class="muted small">rank ${d.rank} of 12 · ${(d.probability * 100).toFixed(2)}% in 7 days · ${fmt(d.relative_risk, 2)}× the sport average (${(d.sport_average * 100).toFixed(2)}%)</p></div>${levelBadge(d.level)}</div>
      <div class="grid g2">
        <div><h3 style="margin-bottom:8px">Risk drivers</h3>${divergingBars(d.drivers.drivers)}
          <h3 style="margin:16px 0 8px">Trend</h3>${lineChart({ labels: d.trend.dates.map(shortDate), height: 170, yMin: 0, yFmt: (v) => (v * 100).toFixed(1) + "%", series: [{ name: d.label, color: "var(--ember)", values: d.trend.probability }] })}
          <h3 style="margin:16px 0 8px">Likely injury type</h3><div class="stack">${hbars(types, "var(--sky)")}</div></div>
        <div><h3 style="margin-bottom:8px">What-if (re-run through the model)</h3>
          <div class="table-wrap"><table><thead><tr><th>Scenario</th><th>${esc(REGION_SHORT[d.region])}</th><th>Change</th><th>Any</th></tr></thead><tbody>${cfs}</tbody></table></div>
          <h3 style="margin:16px 0 8px">Prevention</h3><ul class="clean small">${g.prevention.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>
          <h3 style="margin:12px 0 8px">Return to play</h3><p class="small muted">${esc(g.return_to_play)}</p>
          ${g.evidence ? `<p class="small" style="margin-top:8px"><b>Evidence:</b> <span class="muted">${esc(g.evidence)}</span></p>` : ""}
          ${hist || d.previous_injuries_before_tracking ? `<h3 style="margin:12px 0 6px">History</h3><ul class="clean small">${hist}${d.previous_injuries_before_tracking ? `<li>${d.previous_injuries_before_tracking} before tracking started</li>` : ""}</ul>` : ""}
          <p class="disclaimer small">${esc(g.disclaimer)}</p></div>
      </div></div>`;
  }

  function heatmap(h) {
    if (!h.athletes.length) return '<div class="empty">No scored athletes yet.</div>';
    const head = h.regions.map((r) => `<th class="hm-col" title="${esc(r.label)}"><span>${esc(REGION_SHORT[r.region])}</span></th>`).join("");
    const rows = h.athletes.map((a) => `<tr><td class="hm-name"><a href="#/athlete/${a.athlete_id}"><b>${esc(a.name)}</b></a><div class="muted small">${esc(a.sport)} · ${pct(a.overall)} ${badge(a.band)}</div></td>
      ${h.regions.map((r) => { const rr = a.relative_risk[r.region], p = a.probabilities[r.region];
        return `<td class="hm-cell" style="background:${rrColor(rr)}" data-href="#/athlete/${a.athlete_id}?region=${r.region}" title="${esc(a.name)} · ${esc(r.label)}: ${(p * 100).toFixed(2)}% (${rr}× sport avg)">${fmt(rr, 1)}</td>`; }).join("")}</tr>`).join("");
    return `<div class="table-wrap"><table class="heatmap"><thead><tr><th>Athlete</th>${head}</tr></thead><tbody>${rows}</tbody></table></div>
      <p class="muted small" style="margin-top:8px">Cells show 7-day risk relative to the athlete's sport average (1.0 = typical). Click a cell for region details.</p>`;
  }

  function loadCharts(rt) {
    const labels = rt.dates.map(shortDate), L = rt.load;
    return `<div class="grid g2">
      <div class="card"><div class="card-head"><h2>Acute vs chronic load</h2><span class="muted small">EWMA, 7-day / 28-day spans (sRPE)</span></div>${lineChart({ labels, yMin: 0,
        bars: { name: "Daily load", color: "var(--sky)", values: L.daily_load },
        series: [{ name: "Acute (EWMA 7)", color: "var(--ember)", values: L.ewma_acute }, { name: "Chronic (EWMA 28)", color: "var(--mint)", values: L.ewma_chronic }] })}</div>
      <div class="card"><div class="card-head"><h2>Acute : chronic workload ratio</h2><span class="muted small">0.8–1.3 band</span></div>${lineChart({ labels, yMin: 0, yFmt: (v) => fmt(v, 1),
        band: { from: 0.8, to: 1.3, color: "var(--mint)", label: "0.8–1.3" }, series: [{ name: "ACWR (EWMA)", color: "var(--lime)", values: L.acwr }] })}</div>
      <div class="card"><div class="card-head"><h2>Monotony &amp; strain</h2><span class="muted small">Foster: mean/SD of daily load; weekly load × monotony</span></div>${lineChart({ labels, yFmt: (v) => fmt(v, 1),
        band: { from: 0, to: 2, color: "var(--sky)", label: "monotony < 2" }, series: [{ name: "Monotony", color: "var(--ember)", values: L.monotony }] })}
        ${lineChart({ labels, height: 150, yMin: 0, series: [{ name: "Strain", color: "var(--red)", values: L.strain }] })}</div>
      <div class="card"><div class="card-head"><h2>Risk trend</h2><span class="muted small">model run on each day's history</span></div>${lineChart({ labels, yMin: 0, yFmt: (v) => (v * 100).toFixed(0) + "%",
        band: { from: rt.bands.high, to: Math.max(rt.bands.high * 1.6, ...rt.overall), color: "var(--red)", label: "high band" },
        series: [{ name: "Any injury (7-day)", color: "var(--red)", values: rt.overall }] })}
        ${lineChart({ labels, height: 170, yMin: 0, yFmt: (v) => (v * 100).toFixed(1) + "%", series: topRegionSeries(rt) })}</div>
    </div>`;
  }
  const SERIES_COLORS = ["var(--ember)", "var(--sky)", "var(--lime)", "#c792ea"];
  function topRegionSeries(rt) {
    const last = (r) => rt.regions[r][rt.regions[r].length - 1];
    return REGIONS.slice().sort((a, b) => last(b) - last(a)).slice(0, 4).map((r, i) => ({ name: REGION_SHORT[r], color: SERIES_COLORS[i], values: rt.regions[r] }));
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
    const top = a.injury ? a.injury.regions.find((r) => r.region === a.injury.top_regions[0]) : null;
    return `
      <div class="grid g3">
        <div class="card"><div class="row" style="gap:18px">${ring(a.readiness_score, BAND_COLOR[a.readiness_band], "lg", "readiness")}
          <div class="stack"><span class="badge ${a.readiness_band}">${esc(a.readiness_band)}</span><span class="muted small">0-100 composite of recovery, load, performance and wellness.</span></div></div></div>
        <div class="card"><div class="row" style="gap:18px">${ring(a.injury_probability * 400, RISK_COLOR[a.injury_risk], "lg", "7-day injury", (a.injury_probability * 100).toFixed(1) + "%")}
          <div class="stack">${badge(a.injury_risk)}${a.overtraining ? '<span class="badge high">overtraining flag</span>' : ""}
          <span class="muted small">7-day injury probability${top ? ` · top region <b>${esc(REGION_SHORT[top.region])}</b>` : ""}</span></div></div></div>
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
    const [s, hm] = await Promise.all([api("/api/stats"), api("/api/squad/heatmap")]);
    const total = Math.max(1, s.athlete_count);
    const stacked = ["high", "moderate", "low", "none"].map((k) => `<i style="width:${(s.risk_bands[k] / total) * 100}%;background:${RISK_COLOR[k]}" title="${k}: ${s.risk_bands[k]}"></i>`).join("");
    const watch = s.watchlist.map((w) => `
      <tr class="click" data-href="#/athlete/${w.athlete_id}"><td><b>${esc(w.name)}</b><div class="muted small">${esc(w.sport)}</div></td>
      <td>${badge(w.injury_risk)}${w.overtraining ? ' <span class="badge high">OT</span>' : ""}</td><td class="mono">${(w.injury_probability * 100).toFixed(1)}%</td>
      <td class="small">${esc(REGION_SHORT[w.top_region] || "—")}</td>
      <td class="mono" style="color:${readinessColor(w.readiness_score)}">${fmt(w.readiness_score)}</td></tr>`).join("");
    const recent = s.recent.map((r) => `
      <tr class="click" data-href="#/athlete/${r.athlete_id}"><td>${shortDate(r.date)}</td><td>${esc(r.name)}</td>
      <td class="mono">${fmt(r.duration_min)}′ · RPE ${fmt(r.rpe, 1)}</td><td>${badge(r.injury_risk)}</td></tr>`).join("");
    const sports = hbars(s.sports.map((x) => ({ label: x.sport, value: x.count, text: String(x.count) })), "var(--mint)");
    $app.innerHTML = `
      <div class="page-head"><div><h1>Squad overview</h1><p class="muted">Who is ready, who is quietly overreaching, which body region is at risk, and what to change this week.</p></div>
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
      <div class="card" style="margin-top:16px"><div class="card-head"><h2>Squad injury heatmap</h2><span class="muted small">athletes × body regions</span></div>${heatmap(hm)}</div>
      <div class="grid g3" style="margin-top:16px">
        <div class="card span2"><div class="card-head"><h2>Watchlist</h2><span class="muted small">sorted by band, then probability</span></div>
          ${watch ? `<div class="table-wrap"><table><thead><tr><th>Athlete</th><th>Band</th><th>7-day p</th><th>Top region</th><th>Readiness</th></tr></thead><tbody>${watch}</tbody></table></div>` : '<div class="empty">No scored sessions yet.</div>'}</div>
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
        <label class="field">Growth last 6 mo (cm)<input name="growth_cm" type="number" min="0" max="20" step="0.5" placeholder="youth only"></label>
      </div>
      <fieldset class="checks"><legend class="muted small">Injured before (regions)</legend>${REGIONS.map((r) => `<label><input type="checkbox" name="injury_history" value="${r}"> ${REGION_SHORT[r]}</label>`).join("")}</fieldset>
      <div class="row small"><label><input type="checkbox" name="nordic_program" value="1"> Does Nordic hamstring programme</label><label><input type="checkbox" name="adductor_program" value="1"> Does Copenhagen adductor programme</label></div>
      <label class="field">Notes<textarea name="notes" rows="2" maxlength="1000"></textarea></label>
      <div class="row"><span class="spacer"></span><button type="button" class="btn ghost" data-close>Cancel</button><button class="btn primary">Save athlete</button></div>
    </form>`;
  }

  const formData = (form, numeric, lists = ["injury_history"], bools = ["nordic_program", "adductor_program"]) => {
    const out = {};
    lists.forEach((k) => { if (form.querySelector(`[name="${k}"]`)) out[k] = []; });
    bools.forEach((k) => { if (form.querySelector(`[name="${k}"]`)) out[k] = false; });
    new FormData(form).forEach((v, k) => {
      if (v === "") return;
      if (lists.includes(k)) out[k].push(v);
      else if (bools.includes(k)) out[k] = true;
      else out[k] = numeric.includes(k) ? Number(v) : v;
    });
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
        <div class="row">${badge(a.latest_risk)}${a.top_region ? `<span class="badge tag">${esc(REGION_SHORT[a.top_region])}</span>` : ""}<span class="spacer"></span><span class="muted small">${a.last_session ? "last " + shortDate(a.last_session) : "no sessions"}</span></div>
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
        const a = await api("/api/athletes", { method: "POST", body: JSON.stringify(formData(e.target, ["age", "height_cm", "weight_kg", "years_training", "previous_injuries", "growth_cm"])) });
        dlg.close(); toast(`${a.name} added`); location.hash = `#/athlete/${a.id}`;
      } catch (err) { toast(err.message, true); }
    };
    if (params.get("q")) { filters.q.focus(); filters.q.setSelectionRange(99, 99); }
  }

  const OPTIONAL_FIELDS = ["sprint_100m_s", "vertical_jump_cm", "asymmetry_pct"];
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

  async function viewAthlete(id, params) {
    const [d, inj, rt, injuries] = await Promise.all([
      api(`/api/athletes/${id}`),
      api(`/api/athletes/${id}/injury-risk?explain=top`).catch(() => null),
      api(`/api/athletes/${id}/risk-trend`).catch(() => null),
      api(`/api/athletes/${id}/injuries`),
    ]);
    const a = d.athlete, t = d.trend, last = d.sessions[d.sessions.length - 1];
    if (inj) inj.sport = a.sport;
    let selected = params.get("region") || (inj ? inj.top_regions[0] : null);
    const rows = [...d.sessions].reverse().map((s) => {
      const p = t.find((x) => x.session_id === s.id) || {};
      return `<tr><td>${shortDate(s.session_date)}</td><td class="small">${s.session_type === "match" ? '<span class="badge tag">match</span>' : esc(s.session_type || "training")}</td><td class="mono">${fmt(s.duration_min)}′</td><td class="mono">${fmt(s.rpe, 1)}</td>
        <td class="mono">${fmt(s.distance_km, 1)}</td><td class="mono">${fmt(s.sprint_100m_s, 2)}</td><td class="mono">${fmt(s.vertical_jump_cm, 1)}</td><td class="mono">${fmt(s.sleep_hours, 1)}</td>
        <td class="mono" style="color:${readinessColor(p.readiness_score)}">${fmt(p.readiness_score)}</td><td>${badge(p.injury_risk)}</td><td class="small">${esc(REGION_SHORT[p.top_region] || "")}</td>
        <td><button class="btn sm ghost danger" data-del-session="${s.id}" title="Delete session">✕</button></td></tr>`;
    }).join("");
    const injRows = injuries.map((i) => `<tr><td>${esc(REGION_SHORT[i.region])}</td><td>${esc(TYPE_SHORT[i.injury_type])}</td><td>${shortDate(i.onset_date)}</td><td>${i.return_date ? shortDate(i.return_date) : "ongoing"}</td>
      <td class="small muted">${esc(i.notes)}</td><td><button class="btn sm ghost danger" data-del-injury="${i.id}" title="Delete">✕</button></td></tr>`).join("");
    const history = (a.injury_history || []).map((r) => REGION_SHORT[r]).join(", ");
    $app.innerHTML = `
      <div class="page-head"><div class="row" style="gap:16px"><div class="avatar" style="width:60px;height:60px;font-size:20px">${initials(a.name)}</div>
        <div><h1>${esc(a.name)}</h1><p class="muted">${esc(a.sport)} · ${esc(a.role)} · ${esc(a.academy)}, ${esc(a.city)}</p></div></div>
        <div class="row"><a class="btn" href="#/lab?athlete=${a.id}">What-if in lab</a><button class="btn primary" id="log">+ Log session</button><button class="btn danger" id="del">Delete</button></div></div>
      <div class="card" style="margin-bottom:16px"><dl class="kv" style="grid-template-columns:repeat(auto-fill,minmax(110px,auto) minmax(60px,1fr))">
        <dt>Age</dt><dd>${a.age}</dd><dt>Sex</dt><dd>${esc(a.sex)}</dd><dt>Height</dt><dd>${fmt(a.height_cm)} cm</dd><dt>Weight</dt><dd>${fmt(a.weight_kg, 1)} kg</dd>
        <dt>Training</dt><dd>${fmt(a.years_training, 1)} y</dd><dt>Prev. injuries</dt><dd>${a.previous_injuries}</dd><dt>Sessions</dt><dd>${a.session_count}</dd>
        <dt>Growth 6 mo</dt><dd>${a.growth_cm ? fmt(a.growth_cm, 1) + " cm" : "—"}</dd><dt>Programmes</dt><dd>${[a.nordic_program && "Nordic", a.adductor_program && "Copenhagen"].filter(Boolean).join(", ") || "none"}</dd>
        <dt>Past regions</dt><dd>${esc(history || "—")}</dd></dl>
        ${a.notes ? `<p class="muted small" style="margin-top:10px">${esc(a.notes)}</p>` : ""}</div>
      <section id="log-wrap" hidden class="card" style="margin-bottom:16px"><div class="card-head"><h2>Log a session</h2><span class="muted small">pre-filled from the last session; leave tests blank if not measured</span></div>
        <form id="session-form" class="stack"><div class="form-grid">
          <label class="field">Date<input type="date" name="session_date" value="${today()}" max="${today()}" required></label>
          <label class="field">Type<select name="session_type"><option>training</option><option>match</option><option>recovery</option></select></label>
          ${SESSION_FIELDS.map(([k, l, mn, mx, st, def]) => { const opt = OPTIONAL_FIELDS.includes(k); const v = opt ? "" : last && last[k] != null ? last[k] : def;
            return `<label class="field">${l}${opt ? " <i>(optional)</i>" : ""}<input type="number" name="${k}" min="${mn}" max="${mx}" step="${st}" value="${v}" ${opt ? "" : "required"}></label>`; }).join("")}
          <label class="field">Hop asymmetry % <i>(optional)</i><input type="number" name="asymmetry_pct" min="0" max="60" step="0.5"></label>
        </div><label class="field">Notes<input name="notes" maxlength="1000" placeholder="optional"></label>
        <div class="row"><span class="spacer"></span><button type="button" class="btn ghost" id="cancel-log">Cancel</button><button class="btn primary">Score &amp; save</button></div></form></section>
      ${inj ? `<h2 class="section-title">Injury risk by body region <span class="muted small">as of ${shortDate(inj.as_of)}</span></h2>${injuryOverview(inj, selected)}<div id="region-slot"><div class="loading">Loading region…</div></div>` : '<div class="empty">No sessions yet. Log the first one to get per-region injury risk, readiness and a 72-hour plan.</div>'}
      ${rt ? `<h2 class="section-title">Load management</h2>${loadCharts(rt)}` : ""}
      ${d.latest ? `<h2 class="section-title">Readiness &amp; plan <span class="muted small">(${shortDate(last.session_date)})</span></h2>${analysisPanel(d.latest)}` : ""}
      ${t.length ? `<div class="grid g2" style="margin-top:16px">
        <div class="card"><div class="card-head"><h2>Readiness &amp; performance</h2></div>${lineChart({ labels: t.map((p) => shortDate(p.date)), yMin: 0, yMax: 100, series: [
          { name: "Readiness", color: "var(--mint)", values: t.map((p) => p.readiness_score) },
          { name: "Performance index", color: "var(--lime)", values: t.map((p) => p.performance_index) }] })}</div>
        <div class="card"><div class="card-head"><h2>Field tests &amp; sleep</h2><span class="muted small">lower 100m is better</span></div>${lineChart({ labels: t.map((p) => shortDate(p.date)), yFmt: (v) => fmt(v, 1), series: [
          { name: "100m (s)", color: "var(--sky)", values: t.map((p) => p.sprint_100m_s) },
          { name: "Sleep (h)", color: "var(--mint)", dash: true, values: t.map((p) => p.sleep_hours) }] })}
          ${lineChart({ labels: t.map((p) => shortDate(p.date)), height: 150, series: [{ name: "Vertical jump (cm)", color: "var(--lime)", values: t.map((p) => p.vertical_jump_cm) }] })}</div>
      </div>` : ""}
      <div class="card" style="margin-top:16px"><div class="card-head"><h2>Injury log</h2><span class="muted small">logged injuries feed the history features and rescore all sessions</span></div>
        ${injRows ? `<div class="table-wrap"><table><thead><tr><th>Region</th><th>Type</th><th>Onset</th><th>Return</th><th>Notes</th><th></th></tr></thead><tbody>${injRows}</tbody></table></div>` : '<div class="empty">No injuries logged.</div>'}
        <form id="injury-form" class="form-grid" style="margin-top:14px">
          <label class="field">Region<select name="region">${REGIONS.map((r) => `<option value="${r}">${REGION_SHORT[r]}</option>`).join("")}</select></label>
          <label class="field">Type<select name="injury_type">${Object.entries(TYPE_SHORT).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</select></label>
          <label class="field">Onset<input type="date" name="onset_date" max="${today()}" required></label>
          <label class="field">Return (optional)<input type="date" name="return_date"></label>
          <label class="field">Notes<input name="notes" maxlength="1000"></label>
          <label class="field">&nbsp;<button class="btn">+ Log injury</button></label>
        </form></div>
      <div class="card" style="margin-top:16px"><div class="card-head"><h2>Session log</h2><span class="muted small">${d.sessions.length} sessions</span></div>
        ${rows ? `<div class="table-wrap"><table><thead><tr><th>Date</th><th>Type</th><th>Dur</th><th>RPE</th><th>km</th><th>100m</th><th>Jump</th><th>Sleep</th><th>Readiness</th><th>Band</th><th>Top region</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>` : '<div class="empty">No sessions.</div>'}</div>`;

    const loadRegion = async (region, scroll) => {
      if (!inj || !region) return;
      selected = region;
      $app.querySelectorAll(".bodymap .region").forEach((g) => g.classList.toggle("sel", g.dataset.region === region));
      $app.querySelectorAll("tr[data-region]").forEach((r) => r.classList.toggle("sel", r.dataset.region === region));
      const slot = document.getElementById("region-slot");
      slot.innerHTML = '<div class="loading">Loading region…</div>';
      try { slot.innerHTML = regionDetail(await api(`/api/athletes/${a.id}/injury-risk/${region}`)); if (scroll) slot.scrollIntoView({ behavior: "smooth", block: "start" }); }
      catch (err) { slot.innerHTML = errorBox(err); }
    };
    $app.querySelectorAll("[data-region]").forEach((el) => {
      const go = (ev) => { ev.preventDefault(); loadRegion(el.dataset.region, true); };
      el.addEventListener("click", go);
      el.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") go(ev); });
    });
    loadRegion(selected, Boolean(params.get("region")));

    const wrap = document.getElementById("log-wrap");
    document.getElementById("log").onclick = () => { wrap.hidden = false; wrap.scrollIntoView({ behavior: "smooth", block: "start" }); };
    document.getElementById("cancel-log").onclick = () => (wrap.hidden = true);
    document.getElementById("session-form").onsubmit = async (e) => {
      e.preventDefault();
      const body = formData(e.target, [...SESSION_FIELDS.map((f) => f[0]), "asymmetry_pct"]);
      body.athlete_id = a.id;
      try {
        const r = await api("/api/sessions", { method: "POST", body: JSON.stringify(body) });
        toast(`Session scored: readiness ${fmt(r.analysis.readiness_score)}, ${r.analysis.injury_risk} risk`);
        route();
      } catch (err) { toast(err.message, true); }
    };
    document.getElementById("injury-form").onsubmit = async (e) => {
      e.preventDefault();
      try { await api(`/api/athletes/${a.id}/injuries`, { method: "POST", body: JSON.stringify(formData(e.target, [])) }); toast("Injury logged; history rescored"); route(); }
      catch (err) { toast(err.message, true); }
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
    $app.querySelectorAll("[data-del-injury]").forEach((b) => (b.onclick = async () => {
      if (!confirm("Delete this injury record?")) return;
      try { await api(`/api/injuries/${b.dataset.delInjury}`, { method: "DELETE" }); toast("Injury deleted"); route(); } catch (err) { toast(err.message, true); }
    }));
  }

  const WHATIF_FIELDS = [
    ["load_change_pct", "This week's load change (%)", -60, 60, 5, 0],
    ["extra_rest_days", "Extra rest days this week", 0, 4, 1, 0],
    ["matches_last_7", "Matches last 7d", 0, 4, 1, 0],
    ["chronic_sessions_per_week", "Typical sessions/week (prev. 3 wks)", 0, 14, 1, 4],
  ];

  async function viewLab(params) {
    const athletes = await api("/api/athletes");
    const sel = params.get("athlete") || "";
    let base = { age: 18, sex: "F", sport: "Athletics", years_training: 3, height_cm: 165, weight_kg: 55, previous_injuries: 0, injury_history: [], nordic_program: false, adductor_program: false, growth_cm: 0 };
    SESSION_FIELDS.forEach(([k, , , , , def]) => (base[k] = def));
    WHATIF_FIELDS.forEach(([k, , , , , def]) => (base[k] = def));
    let athleteName = "";
    if (sel) {
      try {
        const d = await api(`/api/athletes/${sel}`);
        athleteName = d.athlete.name;
        ["age", "sex", "sport", "years_training", "height_cm", "weight_kg", "previous_injuries", "injury_history", "nordic_program", "adductor_program"].forEach((k) => (base[k] = d.athlete[k]));
        base.growth_cm = d.athlete.growth_cm || 0;
        const last = d.sessions[d.sessions.length - 1];
        if (last) SESSION_FIELDS.forEach(([k]) => { if (last[k] != null) base[k] = last[k]; });
      } catch (e) { toast(e.message, true); }
    }
    const slider = ([k, l, mn, mx, st]) => `<label class="slider">${l}<b id="v-${k}">${base[k]}</b><input type="range" name="${k}" min="${mn}" max="${mx}" step="${st}" value="${base[k]}"></label>`;
    $app.innerHTML = `
      <div class="page-head"><div><h1>Analysis lab</h1><p class="muted">Move the sliders to see how load, sleep, rest and prevention work change risk for each body region. Nothing is saved.</p></div>
        <select id="who" style="max-width:280px"><option value="">Guest athlete (custom profile)</option>${athletes.map((a) => `<option value="${a.id}" ${String(a.id) === sel ? "selected" : ""}>${esc(a.name)} — ${esc(a.sport)}</option>`).join("")}</select></div>
      <div class="grid lab-grid" id="lab-grid">
        <form id="lab" class="card stack">
          ${sel ? `<p class="small muted">Using <b>${esc(athleteName)}</b>'s logged calendar for the injury model. The what-if sliders change the last 7 days; sleep applies to the whole week.</p>` : `<p class="small muted">Guest mode builds a 4-week history from these numbers: 3 typical weeks, then this week.</p>`}
          <h2>What-if</h2>${WHATIF_FIELDS.filter(([k]) => !sel || ["load_change_pct", "extra_rest_days"].includes(k)).map(slider).join("")}
          <div class="row small"><label><input type="checkbox" name="nordic_program" value="1" ${base.nordic_program ? "checked" : ""}> Nordic programme</label><label><input type="checkbox" name="adductor_program" value="1" ${base.adductor_program ? "checked" : ""}> Copenhagen adductor</label></div>
          <h2 style="margin-top:6px">Profile</h2>
          <div class="form-grid" style="grid-template-columns:1fr 1fr">
            <label class="field">Sport<select name="sport">${SPORTS.map((s) => `<option ${base.sport === s ? "selected" : ""}>${s}</option>`).join("")}</select></label>
            <label class="field">Sex<select name="sex">${["F", "M", "Other"].map((s) => `<option ${base.sex === s ? "selected" : ""}>${s}</option>`).join("")}</select></label>
            <label class="field">Age<input type="number" name="age" min="10" max="60" value="${base.age}"></label>
            <label class="field">Growth 6 mo (cm)<input type="number" name="growth_cm" min="0" max="20" step="0.5" value="${base.growth_cm}"></label>
            <label class="field">Height (cm)<input type="number" name="height_cm" min="120" max="230" value="${base.height_cm}"></label>
            <label class="field">Weight (kg)<input type="number" name="weight_kg" min="30" max="160" value="${base.weight_kg}"></label>
          </div>
          ${sel ? "" : `<fieldset class="checks"><legend class="muted small">Injured before</legend>${REGIONS.map((r) => `<label><input type="checkbox" name="injury_history" value="${r}" ${base.injury_history.includes(r) ? "checked" : ""}> ${REGION_SHORT[r]}</label>`).join("")}</fieldset>`}
          <input type="hidden" name="years_training" value="${base.years_training}"><input type="hidden" name="previous_injuries" value="${base.previous_injuries}">
          <h2 style="margin-top:6px">Session &amp; week</h2>${SESSION_FIELDS.map(slider).join("")}
          <button type="button" class="btn ghost" id="reset">Reset to baseline</button>
        </form>
        <div id="lab-out"><div class="loading">Scoring…</div></div>
      </div>`;
    const form = document.getElementById("lab"), out = document.getElementById("lab-out");
    const numeric = ["age", "years_training", "height_cm", "weight_kg", "previous_injuries", "growth_cm", ...SESSION_FIELDS.map((f) => f[0]), ...WHATIF_FIELDS.map((f) => f[0])];
    let selected = null;
    const run = debounce(async () => {
      const body = formData(form, numeric);
      [...SESSION_FIELDS, ...WHATIF_FIELDS].forEach(([k]) => { const el = document.getElementById(`v-${k}`); if (el) el.textContent = body[k]; });
      if (sel) { body.athlete_id = Number(sel); delete body.injury_history; }
      if (!body.growth_cm) delete body.growth_cm;
      try {
        const r = await api("/api/analyze", { method: "POST", body: JSON.stringify(body) });
        r.injury.sport = body.sport;
        if (!selected || !r.injury.regions.some((x) => x.region === selected)) selected = r.injury.top_regions[0];
        out.innerHTML = `${injuryOverview(r.injury, selected)}<div style="margin-top:16px">${analysisPanel(r)}</div>`;
        out.querySelectorAll("[data-region]").forEach((el) => el.addEventListener("click", (ev) => { ev.preventDefault(); selected = el.dataset.region; run(); }));
      } catch (e) { out.innerHTML = errorBox(e); }
    }, 220);
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
          <p class="small" style="margin-top:8px"><b>Optional:</b> <span class="muted">sex, role, city, academy, years_training, previous_injuries, injury_history (quoted, e.g. "knee,ankle"), growth_cm, nordic_program, adductor_program, distance_km, sprint_100m_s, vertical_jump_cm, resting_hr, session_hr_avg, rpe, sleep_hours, wellness, sessions_last_7, rest_days_last_7, session_type (training / match / recovery), asymmetry_pct, notes</span></p>
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

  function calibrationPlot(bins, color = "var(--lime)") {
    const W = 320, H = 260, P = { l: 44, r: 10, t: 10, b: 34 };
    const hi = Math.max(...bins.flatMap((b) => [b.pred, b.obs]), 0.01) * 1.08;
    const x = (v) => P.l + (v / hi) * (W - P.l - P.r), y = (v) => H - P.b - (v / hi) * (H - P.t - P.b);
    const ticks = [0, hi / 4, hi / 2, (3 * hi) / 4, hi];
    let svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="calibration plot"><g class="grid">`;
    ticks.forEach((t) => { svg += `<line x1="${P.l}" x2="${W - P.r}" y1="${y(t)}" y2="${y(t)}"/><text x="${P.l - 5}" y="${y(t) + 3}" text-anchor="end">${(t * 100).toFixed(0)}%</text><text x="${x(t)}" y="${H - P.b + 14}" text-anchor="middle">${(t * 100).toFixed(0)}%</text>`; });
    svg += `</g><line x1="${x(0)}" y1="${y(0)}" x2="${x(hi)}" y2="${y(hi)}" stroke="rgba(255,255,255,.25)" stroke-dasharray="4 4"/>`;
    svg += `<polyline fill="none" stroke="${color}" stroke-width="2.2" points="${bins.map((b) => `${x(b.pred)},${y(b.obs)}`).join(" ")}"/>`;
    bins.forEach((b) => { svg += `<circle class="dot" cx="${x(b.pred)}" cy="${y(b.obs)}" r="4" fill="${color}"><title>predicted ${(b.pred * 100).toFixed(1)}% · observed ${(b.obs * 100).toFixed(1)}% · n=${b.n}</title></circle>`; });
    svg += `<text x="${(W + P.l) / 2}" y="${H - 4}" text-anchor="middle">predicted 7-day risk</text><text x="12" y="${H / 2}" transform="rotate(-90 12 ${H / 2})" text-anchor="middle">observed</text></svg>`;
    return svg;
  }

  async function viewModel() {
    const [m, im] = await Promise.all([api("/api/model"), api("/api/injury-model")]);
    const sim = im.simulated, real = im.real, perf = m.metrics.performance || {};
    const g = sim.grouped_cv, tmp = sim.temporal_unseen_athletes;
    const n3 = (v) => (v == null ? "—" : Number(v).toFixed(3));
    const label = (k) => (k === "any" ? "Any injury" : REGION_SHORT[k]);
    const regionRows = ["any", ...REGIONS].map((k) => `<tr${k === "any" ? ' class="strong"' : ""}><td>${esc(label(k))}</td><td class="mono">${(sim.label_prevalence[k] * 100).toFixed(2)}%</td>
      <td class="mono"><b>${n3(g[k].model.roc_auc)}</b></td><td class="mono muted">${n3(g[k].baseline_sport_rate.roc_auc)}</td>
      <td class="mono"><b>${n3(g[k].model.pr_auc)}</b></td><td class="mono muted">${n3(g[k].baseline_sport_rate.pr_auc)}</td>
      <td class="mono">${g[k].model.brier.toFixed(4)}</td><td class="mono muted">${g[k].baseline_prevalence.brier.toFixed(4)}</td>
      <td class="mono">${n3(tmp[k].model.roc_auc)}</td><td class="mono">${g[k].calibration.ece.toFixed(4)}</td></tr>`).join("");
    const anyRows = [["Calibrated HGB (this model)", g.any.model], ["Prevalence (constant)", g.any.baseline_prevalence], ["Sport injury rate", g.any.baseline_sport_rate],
      ["ACWR-only logistic", g.any.baseline_acwr_logistic], ["Logistic regression, all features", g.any.baseline_logistic_all_features], ["Oracle: simulator's true hazard (ceiling)", g.any.oracle_true_hazard]]
      .map(([n, v], i) => `<tr${i === 0 ? ' class="strong"' : ""}><td>${esc(n)}</td><td class="mono">${n3(v.roc_auc)}</td><td class="mono">${v.pr_auc == null ? "—" : Number(v.pr_auc).toFixed(4)}</td><td class="mono">${v.brier.toFixed(5)}</td></tr>`).join("");
    const tk = sim.top_k_region;
    const permKeys = Object.keys(sim.permutation_importance_auc_drop);
    const perm = (k) => hbars(Object.entries(sim.permutation_importance_auc_drop[k]).filter(([, v]) => v > 0).slice(0, 10).map(([gname, v]) => ({ label: gname.replace(/_/g, " "), value: v, text: v.toFixed(3) })));
    const ra = real.app_features, rb = real.all_features_benchmark;
    const realRow = (n, v) => `<tr><td>${esc(n)}</td><td class="mono">${n3(v.roc_auc)}</td><td class="mono">${v.pr_auc == null ? "—" : Number(v.pr_auc).toFixed(4)}</td><td class="mono">${v.brier.toFixed(5)}</td></tr>`;
    const kmMax = Math.max(...real.injury_rate_by_km_ratio.map((b) => b.rate || 0));
    $app.innerHTML = `
      <div class="page-head"><div><h1>Model card</h1><p class="muted">Every number below was produced by the training script on held-out athletes (injury models trained ${esc(im.trained_at)}, scikit-learn ${esc(im.sklearn_version)}, ${esc(im.train_seconds)} s).</p></div></div>
      <div class="card honest"><h2 style="margin-bottom:8px">What is real and what is simulated</h2>
        <div class="grid g2"><div><span class="badge tag">simulated</span><p class="small" style="margin-top:6px">The <b>12 per-region models</b>, the overall model and the injury-type model are trained on <b>${sim.n_athletes.toLocaleString()} simulated athlete-seasons</b> (${sim.season_days} days each, ${sim.rows.toLocaleString()} athlete-days, ${sim.injuries.toLocaleString()} injuries, ${sim.injuries_per_athlete_season}/athlete-season). No openly downloadable dataset with body-region injury labels and daily load was found. The simulator's region shares and risk factors follow the published studies listed below where we read them; the strength of each effect is partly assumed. These metrics measure how well the model recovers the simulator, <b>not real-world accuracy</b>.</p></div>
        <div><span class="badge low">real</span><p class="small" style="margin-top:6px">The <b>real-data cross-check</b> is trained on ${real.source.rows.toLocaleString()} athlete-days from ${real.source.athletes} competitive runners (Lövdal, den Hartigh &amp; Azzopardi 2021, <a href="https://doi.org/10.34894/UWU9PV" target="_blank" rel="noopener">doi:10.34894/UWU9PV</a>, ${esc(real.source.licence)}). Labels are binary (${real.source.positives} injury days, ${(real.source.prevalence * 100).toFixed(2)}%), with no body region, so this model cannot say <i>where</i>. It checks the load features on real data.</p></div></div></div>

      <div class="card" style="margin-top:16px"><div class="card-head"><h2>Per-region performance</h2><span class="muted small">5-fold athlete-grouped CV (no athlete in both train and test) · temporal = unseen athletes, last third of season</span></div>
        <div class="table-wrap"><table class="metrics"><thead><tr><th>Target</th><th>Prevalence</th><th>ROC AUC</th><th>Sport-rate AUC</th><th>PR AUC</th><th>Sport-rate PR</th><th>Brier</th><th>Const. Brier</th><th>Temporal AUC</th><th>ECE</th></tr></thead><tbody>${regionRows}</tbody></table></div>
        <p class="muted small" style="margin-top:8px">Label: injury onset in the region within the next 7 days. ROC AUC 0.5 = chance. PR AUC should be compared with prevalence. ECE = expected calibration error (lower is better).</p></div>

      <div class="grid g2" style="margin-top:16px">
        <div class="card"><div class="card-head"><h2>Any injury vs baselines</h2></div>
          <div class="table-wrap"><table><thead><tr><th>Model</th><th>ROC AUC</th><th>PR AUC</th><th>Brier</th></tr></thead><tbody>${anyRows}</tbody></table></div>
          <p class="muted small" style="margin-top:8px">The oracle knows each simulated athlete's hidden frailty and future plan. It is the best any model could do on this data, which shows how much is irreducible noise.</p>
          <h3 style="margin:16px 0 8px">Which region? (windows with an injury)</h3>
          <div class="table-wrap"><table><thead><tr><th></th><th>Model</th><th>Sport prior</th><th>Global prior</th></tr></thead><tbody>
            <tr><td>Top-1 region correct</td><td class="mono"><b>${pct(tk.top1.model)}</b></td><td class="mono">${pct(tk.top1.sport_prior)}</td><td class="mono">${pct(tk.top1.global_prior)}</td></tr>
            <tr><td>Top-3 regions contain it</td><td class="mono"><b>${pct(tk.top3.model)}</b></td><td class="mono">${pct(tk.top3.sport_prior)}</td><td class="mono">${pct(tk.top3.global_prior)}</td></tr></tbody></table></div>
          <h3 style="margin:16px 0 8px">Injury-type model</h3><dl class="kv"><dt>Events</dt><dd>${sim.injury_type.n_events.toLocaleString()}</dd><dt>Accuracy</dt><dd>${pct(sim.injury_type.model_accuracy)} <span class="muted small">(region-majority baseline ${pct(sim.injury_type.baseline_region_majority_accuracy)})</span></dd><dt>Log loss</dt><dd>${sim.injury_type.model_log_loss} <span class="muted small">(region type-frequency baseline ${sim.injury_type.baseline_region_frequency_log_loss ?? "—"})</span></dd></dl>
          <p class="muted small">Injury type is mostly determined by the region; load context adds little beyond that.</p></div>
        <div class="card"><div class="card-head"><h2>Calibration (any injury)</h2><span class="muted small">out-of-fold, decile bins · ECE ${g.any.calibration.ece}</span></div>
          ${calibrationPlot(g.any.calibration.bins)}
          <div class="row" style="margin-top:8px"><label class="field" style="flex:1">Region<select id="cal-region">${["any", ...REGIONS].map((k) => `<option value="${k}">${esc(label(k))}</option>`).join("")}</select></label></div>
          <div id="cal-slot"></div></div>
      </div>

      <div class="grid g2" style="margin-top:16px">
        <div class="card"><div class="card-head"><h2>Feature-group importance</h2><select id="perm-target" style="max-width:180px">${permKeys.map((k) => `<option value="${k}">${esc(label(k))}</option>`).join("")}</select></div>
          <p class="muted small" style="margin-bottom:8px">Grouped permutation importance: drop in ROC AUC on the temporal test set when a feature group is shuffled.</p><div class="stack" id="perm-slot">${perm(permKeys[0])}</div></div>
        <div class="card"><div class="card-head"><h2>Real data: competitive runners</h2><span class="badge low">real</span></div>
          <div class="table-wrap"><table><thead><tr><th>Model (real data)</th><th>ROC AUC</th><th>PR AUC</th><th>Brier</th></tr></thead><tbody>
            ${realRow("App-mappable features, athlete-grouped CV", ra.grouped_cv)}${realRow("App-mappable features, temporal", ra.temporal)}
            ${realRow("All 69 dataset features, grouped CV (benchmark)", rb.grouped_cv)}${realRow("All features, temporal", rb.temporal)}
            ${realRow("Prevalence baseline", real.baseline_prevalence)}</tbody></table></div>
          <h3 style="margin:14px 0 8px">Injury rate by last-week km vs 3-week average</h3>
          <div class="stack">${real.injury_rate_by_km_ratio.filter((b) => b.rate != null).map((b) => `<div class="metric" style="grid-template-columns:80px 1fr 90px"><span class="muted">${esc(b.range)}×</span><div class="bar"><i style="width:${(b.rate / kmMax) * 100}%;background:var(--ember)"></i></div><span>${(b.rate * 100).toFixed(2)}% <span class="muted small">n=${b.n.toLocaleString()}</span></span></div>`).join("")}</div>
          <p class="muted small" style="margin-top:8px">${esc(ra.temporal_note)}. The real-data signal is weak, which matches the original paper's modest results. Injury prediction from training logs alone is hard.</p></div>
      </div>

      <div class="grid g2" style="margin-top:16px">
        <div class="card"><div class="card-head"><h2>Performance index</h2><span class="badge tag">regressor</span></div><p class="muted small" style="margin-bottom:10px">${esc(m.performance_model)}</p>
          <dl class="kv"><dt>Held-out MAE</dt><dd>${perf.holdout_mae} <span class="muted small">(predict-the-mean ${perf.baseline_mean_mae})</span></dd><dt>Held-out R²</dt><dd>${perf.holdout_r2}</dd><dt>Data</dt><dd>synthetic cohort</dd></dl>
          <h3 style="margin:16px 0 6px">Design</h3><p class="muted small">${esc(im.design)}</p>
          <h3 style="margin:16px 0 6px">Bands</h3><dl class="kv">${Object.entries(m.risk_bands).map(([k, v]) => `<dt>${k === "region_levels" ? "regions" : badge(k)}</dt><dd class="small">${esc(v)}</dd>`).join("")}</dl></div>
        <div class="card"><h2 style="margin-bottom:10px">Limitations</h2><ul class="clean small">${im.limitations.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>
          <h3 style="margin:16px 0 6px">Sources actually used</h3><ul class="clean small sources">${im.sources.map((s) => { const url = s.url || (s.pmid ? `https://pubmed.ncbi.nlm.nih.gov/${s.pmid}/` : s.doi ? `https://doi.org/${s.doi}` : null); const cite = `${s.key}. ${s.title}. ${s.journal || ""}`;
            return `<li>${url ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(cite)}</a>` : esc(cite)}${s.licence ? ` <span class="badge low">${esc(s.licence)}</span>` : ""}${s.used_for ? ` <span class="muted">· ${esc(s.used_for)}</span>` : ""}</li>`; }).join("")}</ul></div>
      </div>`;
    const calSel = document.getElementById("cal-region"), calSlot = document.getElementById("cal-slot");
    calSel.onchange = () => { const k = calSel.value; calSlot.innerHTML = k === "any" ? "" : `<h3 style="margin:8px 0">${esc(label(k))} · ECE ${g[k].calibration.ece}</h3>${calibrationPlot(g[k].calibration.bins, "var(--ember)")}`; };
    document.getElementById("perm-target").onchange = (e) => (document.getElementById("perm-slot").innerHTML = perm(e.target.value));
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
      else if (name === "athlete" && parts[1]) await viewAthlete(encodeURIComponent(parts[1]), params);
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
