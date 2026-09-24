/* ============================================================
   DYNAMIC ORACLE — frontend application
   Match Lab · Tournament · Champion Model
   ============================================================ */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[c]));
const fmtPct = (v) => (v == null ? "—" : Number(v).toFixed(1) + "%");
const fmtInt = (v) => Number(v).toLocaleString("en-US");

let toastTimer = null;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 4200);
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) {
    const detail = data && data.detail ? data.detail : `HTTP ${res.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

/* ---------------- tabs ---------------- */
$$(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    $$(".view").forEach((v) => v.classList.remove("active"));
    $("#view-" + btn.dataset.view).classList.add("active");
    if (btn.dataset.view === "champion") loadChampion();
    if (btn.dataset.view === "tournament") loadPresets();
  });
});

/* ---------------- engine status ---------------- */
(async function checkStatus() {
  try {
    const meta = await api("/api/meta");
    const pill = $("#engineStatus");
    pill.classList.add("online");
    pill.innerHTML = `<span class="dot"></span> ENGINE ONLINE · ${fmtInt(meta.total_teams_indexed)} TEAMS`;
    window.ORACLE_YEARS = meta.available_years;
    window.ORACLE_FORMATIONS = meta.supported_formations;
  } catch (e) {
    $("#engineStatus").innerHTML = `<span class="dot"></span> OFFLINE — start the server`;
  }
})();
/* ============================================================
   Team picker (year + league + type + searchable combobox)
   ============================================================ */
const leaguesCache = {};
const teamsCache = {};

async function getLeagues(year) {
  if (!leaguesCache[year]) leaguesCache[year] = await api(`/api/leagues?year=${year}`);
  return leaguesCache[year];
}

async function getTeams(year, q = "") {
  const key = `${year}|${q.toLowerCase()}`;
  if (!teamsCache[key]) {
    teamsCache[key] = await api(`/api/teams?year=${year}&type=all&q=${encodeURIComponent(q)}&limit=400`);
  }
  return teamsCache[key].teams;
}

function createPicker(mountId, opts = {}) {
  const mount = $("#" + mountId);
  mount.innerHTML = `
    <div class="picker-row">
      <select class="py-year"></select>
      <select class="py-league"><option value="">All leagues…</option></select>
    </div>
    <div class="picker-row">
      <select class="py-type">
        <option value="all">Clubs + Nations</option>
        <option value="club">Clubs only</option>
        <option value="national">Nations only</option>
      </select>
      <div class="combo">
        <input type="text" class="py-search" placeholder="Search team…" autocomplete="off" />
        <div class="combo-list"></div>
      </div>
    </div>`;
  // pass onSelect hook
  const state = {
    el: mount, year: 2022, league: "", type: "all", team: null,
    onSelect: opts.onSelect || (() => {}),
    defaultTeam: opts.defaultTeam || null,
    defaultYear: opts.defaultYear || 2022,
  };
  state.yearSelect = $(".py-year", mount);
  state.leagueSelect = $(".py-league", mount);
  state.typeSelect = $(".py-type", mount);
  state.search = $(".py-search", mount);
  state.list = $(".combo-list", mount);

  const years = window.ORACLE_YEARS || [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2026];
  years.forEach((y) => {
    const o = document.createElement("option");
    o.value = y;
    o.textContent = y === 2026 ? "WC 2026" : y;
    if (y === state.defaultYear) o.selected = true;
    state.yearSelect.appendChild(o);
  });
  state.year = state.defaultYear;

  let debounce = null;
  async function refreshLeagues() {
    try {
      const data = await getLeagues(state.year);
      const cur = state.leagueSelect.value;
      state.leagueSelect.innerHTML = `<option value="">All leagues…</option>`;
      data.leagues.forEach((l) => {
        const o = document.createElement("option");
        o.value = l.name;
        o.textContent = `${l.name} (${l.team_count})`;
        state.leagueSelect.appendChild(o);
      });
      if ([...state.leagueSelect.options].some((o) => o.value === cur)) {
        state.leagueSelect.value = cur;
      }
      state.league = state.leagueSelect.value;
    } catch (e) { /* leagues optional */ }
  }

  async function renderList(query) {
    const teams = await getTeams(state.year, query);
    const filtered = teams.filter((t) => {
      if (state.type !== "all" && t.type !== state.type) return false;
      if (state.league && (t.league || "Other") !== state.league) return false;
      return true;
    });
    if (!filtered.length) {
      state.list.innerHTML = `<div class="combo-empty">No teams found</div>`;
    } else {
      state.list.innerHTML = filtered.slice(0, 60).map((t, i) => `
        <div class="combo-item${i === 0 ? " hl" : ""}" data-i="${i}">
          <span class="ci-name">${esc(t.name)}</span>
          <span class="ci-meta">${t.type === "club" ? "♣" : "⚑"} ${t.avg_rating} · ${esc(t.league || "")}</span>
        </div>`).join("");
      state.list._items = filtered.slice(0, 60);
    }
    state.list.classList.add("open");
  }

  function selectTeam(team) {
    state.team = { name: team.name, year: state.year };
    state.search.value = team.name;
    state.list.classList.remove("open");
    state.onSelect(state.team);
  }

  state.search.addEventListener("focus", () => renderList(state.search.value.trim()));
  state.search.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => renderList(state.search.value.trim()), 220);
  });
  state.search.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && state.list._items && state.list._items.length) {
      selectTeam(state.list._items[0]);
    }
    if (e.key === "Escape") state.list.classList.remove("open");
  });
  state.list.addEventListener("click", (e) => {
    const item = e.target.closest(".combo-item");
    if (item && state.list._items) selectTeam(state.list._items[+item.dataset.i]);
  });
  document.addEventListener("click", (e) => {
    if (!mount.contains(e.target)) state.list.classList.remove("open");
  });
  state.yearSelect.addEventListener("change", async () => {
    state.year = +state.yearSelect.value;
    state.team = null;
    state.search.value = "";
    await refreshLeagues();
    renderList("");
  });
  state.leagueSelect.addEventListener("change", () => {
    state.league = state.leagueSelect.value;
    renderList(state.search.value.trim());
  });
  state.typeSelect.addEventListener("change", () => {
    state.type = state.typeSelect.value;
    renderList(state.search.value.trim());
  });

  state.setTeam = async (name, year) => {
    state.yearSelect.value = year;
    state.year = year;
    await refreshLeagues();
    state.team = { name, year };
    state.search.value = name;
    state.onSelect(state.team);
  };

  refreshLeagues().then(() => {
    if (state.defaultTeam) state.setTeam(state.defaultTeam, state.defaultYear);
  });
  return state;
}
/* ============================================================
   MATCH LAB
   ============================================================ */
const FORMATIONS = ["4-3-3", "4-2-3-1", "3-5-2", "4-4-2", "5-3-2"];
["formA", "formB"].forEach((id) => {
  const sel = $("#" + id);
  FORMATIONS.forEach((f) => {
    const o = document.createElement("option");
    o.value = f; o.textContent = f;
    sel.appendChild(o);
  });
});

async function loadTeamMeta(team, metaElId, resolvedElId) {
  const box = $("#" + metaElId);
  if (!team) { box.innerHTML = `<p class="placeholder">Pick a team to inspect its squad ratings.</p>`; return; }
  box.innerHTML = `<p class="placeholder">Loading squad…</p>`;
  try {
    const p = await api(`/api/team-preview?team=${encodeURIComponent(team.name)}&year=${team.year}`);
    $("#" + resolvedElId).textContent = p.year_resolved !== team.year ? `(plays as ${p.year_resolved})` : "";
    const chips = [
      ["OVR", p.overall], ["ATT", p.attack], ["MID", p.midfield],
      ["DEF", p.defence], ["GK", p.gk], ["CHEM", p.chemistry],
    ].map(([k, v]) => `<span class="rchip">${k}<b>${v}</b></span>`).join("");
    const stars = (p.top_players || []).slice(0, 3).map((s) => `${esc(s.name)} <b>${s.overall}</b>`).join(" · ");
    box.innerHTML = `<div class="rating-chips">${chips}</div><p class="stars-line">★ ${stars}</p>`;
  } catch (e) {
    box.innerHTML = `<p class="placeholder">Preview unavailable: ${esc(e.message)}</p>`;
  }
}

const pickerA = createPicker("pickerA", {
  defaultTeam: "FC Barcelona", defaultYear: 2015,
  onSelect: (t) => loadTeamMeta(t, "metaA", "resolvedA"),
});
const pickerB = createPicker("pickerB", {
  defaultTeam: "Morocco", defaultYear: 2026,
  onSelect: (t) => loadTeamMeta(t, "metaB", "resolvedB"),
});

$("#simCount").addEventListener("input", (e) => {
  $("#simCountLabel").textContent = fmtInt(+e.target.value);
});

$("#presetChips").addEventListener("click", async (e) => {
  const chip = e.target.closest(".preset-chip");
  if (!chip) return;
  await pickerA.setTeam(chip.dataset.a, +chip.dataset.ya);
  await pickerB.setTeam(chip.dataset.b, +chip.dataset.yb);
});

(async function loadMatchPresets() {
  try {
    const data = await api("/api/presets");
    $("#presetChips").innerHTML = data.presets.map((p) => `
      <button class="preset-chip" data-a="${esc(p.team_a)}" data-ya="${p.year_a}"
        data-b="${esc(p.team_b)}" data-yb="${p.year_b}" title="${esc(p.tag)}">${esc(p.title)}</button>`).join("");
  } catch { /* presets optional */ }
})();

function animateBars(scope) {
  requestAnimationFrame(() => requestAnimationFrame(() => {
    $$("[data-w]", scope).forEach((el) => { el.style.width = el.dataset.w + "%"; });
  }));
}
$("#runMatch").addEventListener("click", async () => {
  if (!pickerA.team || !pickerB.team) { toast("Select both teams first."); return; }
  const btn = $("#runMatch");
  btn.disabled = true;
  btn.textContent = "⏳ SIMULATING…";
  const box = $("#matchResults");
  box.classList.remove("hidden");
  box.innerHTML = `<div class="progress-note">Running Monte Carlo simulation — sampling scorelines from the Dixon-Coles xG model…</div>`;
  try {
    const res = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        team_a: pickerA.team.name, year_a: pickerA.team.year,
        team_b: pickerB.team.name, year_b: pickerB.team.year,
        n_simulations: +$("#simCount").value,
        neutral: $("#neutralVenue").checked,
        formation_a: $("#formA").value, formation_b: $("#formB").value,
      }),
    });
    stopReplay();
    renderMatchResults(res);
  } catch (e) {
    box.innerHTML = `<div class="progress-note">Simulation failed: ${esc(e.message)}</div>`;
  }
  btn.disabled = false;
  btn.textContent = "⚡ SIMULATE MATCH";
});

function h2hAttr(label, va, vb) {
  const max = Math.max(va, vb, 1);
  return `<div class="h2h-row">
    <div class="h2h-label"><span>${label}</span></div>
    <div class="h2h-bars"><div class="hb ha"><div data-w="${(va / max * 100).toFixed(1)}"></div></div>
    <div class="hb hb2"><div data-w="${(vb / max * 100).toFixed(1)}"></div></div></div>
    <div class="h2h-vals"><span>${va}</span><span>${vb}</span></div>
  </div>`;
}
function renderMatchResults(res) {
  const box = $("#matchResults");
  const p = res.probabilities, A = res.team_a, B = res.team_b;
  const maxScore = Math.max(...res.top_scorelines.map((s) => s.pct), 1);

  box.innerHTML = `
  <div class="card">
    <h3>🎯 Win Probability — ${fmtInt(res.matchup.n_simulations)} Monte Carlo runs</h3>
    <div class="prob-head">
      <span class="ph-team">${esc(A.name)} (${A.year_resolved})</span>
      <span class="ph-team">${esc(B.name)} (${B.year_resolved})</span>
    </div>
    <div class="tribar">
      <div class="pa" data-w="${p.team_a_win_pct}">${p.team_a_win_pct}%</div>
      <div class="pd" data-w="${p.draw_pct}">${p.draw_pct}%</div>
      <div class="pb" data-w="${p.team_b_win_pct}">${p.team_b_win_pct}%</div>
    </div>
    <div class="prob-labels"><span>● ${esc(A.name)} win</span><span>● Draw</span><span>${esc(B.name)} win ●</span></div>
  </div>

  <div class="stat-cards">
    <div class="stat-card"><div class="sc-label">Predicted score</div><div class="sc-value gold">${esc(res.most_likely_scoreline)}</div></div>
    <div class="stat-card"><div class="sc-label">Confidence</div><div class="sc-value green">${res.confidence_pct}%</div></div>
    <div class="stat-card"><div class="sc-label">Expected goals (xG)</div><div class="sc-value blue">${res.xg.team_a} – ${res.xg.team_b}</div></div>
    <div class="stat-card"><div class="sc-label">Total xG</div><div class="sc-value">${res.xg.total}</div></div>
  </div>

  <div class="res-grid-2">
    <div class="card">
      <h3>📊 Most Likely Scorelines</h3>
      ${res.top_scorelines.map((s) => `
        <div class="scoreline-row">
          <span class="sl">${esc(s.scoreline)}</span>
          <div class="bar-track"><div class="bar-fill" data-w="${(s.pct / maxScore * 100).toFixed(1)}"></div></div>
          <span class="pct">${s.pct}%</span>
        </div>`).join("")}
    </div>
    <div class="card">
      <h3>💰 Betting Markets</h3>
      <div class="market-row"><span>Over 2.5 goals</span><b>${res.extra_markets.over_2_5_goals_pct}%</b></div>
      <div class="market-row"><span>Under 2.5 goals</span><b>${res.extra_markets.under_2_5_goals_pct}%</b></div>
      <div class="market-row"><span>Both teams to score</span><b>${res.extra_markets.btts_pct}%</b></div>
      <div class="market-row"><span>Pens — ${esc(A.name)}</span><b>${res.extra_markets.knockout_penalties.team_a_pen_win_pct}%</b></div>
      <div class="market-row"><span>Pens — ${esc(B.name)}</span><b>${res.extra_markets.knockout_penalties.team_b_pen_win_pct}%</b></div>
    </div>
  </div>

  <div class="res-grid-2">
    <div class="card">
      <h3>⚖️ Squad Head-to-Head</h3>
      ${h2hAttr("Overall", A.rating.overall, B.rating.overall)}
      ${h2hAttr("Attack", A.rating.attack, B.rating.attack)}
      ${h2hAttr("Midfield", A.rating.midfield, B.rating.midfield)}
      ${h2hAttr("Defence", A.rating.defence, B.rating.defence)}
      ${h2hAttr("Goalkeeper", A.rating.gk, B.rating.gk)}
      ${h2hAttr("Pace", A.rating.pace, B.rating.pace)}
    </div>
    <div class="card">
      <h3>🧪 Model Diagnostics</h3>
      <div class="market-row"><span>RPS</span><b>${res.evaluation_metrics.headline_rps}</b></div>
      <div class="market-row"><span>Log loss</span><b>${res.evaluation_metrics.log_loss}</b></div>
      <div class="market-row"><span>Brier score</span><b>${res.evaluation_metrics.brier_score}</b></div>
      <div class="market-row"><span>Champion accuracy</span><b>60.14%</b></div>
    </div>
  </div>

  <div class="card">
    <h3>🎬 Simulated Match Replay</h3>
    <div class="replay-wrap">
      <div>
        <div class="pitch" id="pitch">
          <div class="pitch-flash" id="pitchFlash"></div>
          <div class="pitch-score" id="pitchScore">0 – 0</div>
          <div class="pitch-clock" id="pitchClock">0'</div>
          <div class="pitch-event" id="pitchEvent"></div>
        </div>
        <div class="replay-controls">
          <button id="replayPlay">▶ PLAY</button>
          <button id="replayReset">↺ RESET</button>
          <div class="replay-progress"><div id="replayBar"></div></div>
        </div>
      </div>
      <div class="event-feed" id="eventFeed"></div>
    </div>
  </div>

  <div class="card">
    <h3>📋 Starting Lineups</h3>
    <div class="lineup-cols">
      <div class="lineup-team"><h4>${esc(A.name)} · ${esc(A.formation)}</h4>
        ${A.lineup.map((pl) => `<div class="lineup-player"><span><span class="pos">${esc(pl.slot_label)}</span>${esc(pl.name)}</span><span class="ovr">${pl.overall}</span></div>`).join("")}
      </div>
      <div class="lineup-team"><h4>${esc(B.name)} · ${esc(B.formation)}</h4>
        ${B.lineup.map((pl) => `<div class="lineup-player"><span><span class="pos">${esc(pl.slot_label)}</span>${esc(pl.name)}</span><span class="ovr">${pl.overall}</span></div>`).join("")}
      </div>
    </div>
  </div>`;

  animateBars(box);
  initReplay(res.timeline || []);
  box.scrollIntoView({ behavior: "smooth", block: "start" });
}
/* ---------------- match replay player ---------------- */
let replayTimer = null;
let replayEvents = [];
let replayIdx = 0;

function stopReplay() {
  if (replayTimer) { clearInterval(replayTimer); replayTimer = null; }
  const btn = $("#replayPlay");
  if (btn) btn.textContent = "▶ PLAY";
}

function initReplay(events) {
  replayEvents = (events || []).slice().sort((a, b) => (a.minute || 0) - (b.minute || 0));
  replayIdx = 0;
  const feed = $("#eventFeed");
  feed.innerHTML = replayEvents.map((ev, i) => {
    const isGoal = (ev.type || "").startsWith("goal");
    return `<div class="event-item${isGoal ? " goal" : ""}" data-i="${i}">
      <span class="min">${ev.minute}'</span>
      <span>${isGoal ? "⚽ " : ""}${esc(ev.description)} <span class="ev-score">${esc(ev.score || "")}</span></span>
    </div>`;
  }).join("");
  resetReplayUI();
  $("#replayPlay").addEventListener("click", () => {
    if (replayTimer) { stopReplay(); return; }
    if (replayIdx >= replayEvents.length) resetReplayUI();
    $("#replayPlay").textContent = "⏸ PAUSE";
    replayTimer = setInterval(stepReplay, 1100);
    stepReplay();
  });
  $("#replayReset").addEventListener("click", () => { stopReplay(); resetReplayUI(); });
}

function resetReplayUI() {
  replayIdx = 0;
  $("#pitchScore").textContent = "0 – 0";
  $("#pitchClock").textContent = "0'";
  $("#replayBar").style.width = "0%";
  const pe = $("#pitchEvent");
  pe.classList.remove("show");
  $$("#eventFeed .event-item").forEach((el) => el.classList.remove("reached"));
}

function stepReplay() {
  if (replayIdx >= replayEvents.length) { stopReplay(); return; }
  const ev = replayEvents[replayIdx];
  const isGoal = (ev.type || "").startsWith("goal");
  $("#pitchClock").textContent = `${ev.minute}'`;
  if (ev.score) $("#pitchScore").textContent = String(ev.score).replace(" - ", " – ");
  const pe = $("#pitchEvent");
  pe.innerHTML = `<b>${ev.minute}'</b> — ${esc(ev.description)}`;
  pe.classList.toggle("goal", isGoal);
  pe.classList.add("show");
  if (isGoal) {
    const flash = $("#pitchFlash");
    flash.classList.remove("goal-flash");
    void flash.offsetWidth;
    flash.classList.add("goal-flash");
  }
  const item = $(`#eventFeed .event-item[data-i="${replayIdx}"]`);
  if (item) { item.classList.add("reached"); item.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
  $("#replayBar").style.width = ((replayIdx + 1) / replayEvents.length * 100).toFixed(1) + "%";
  replayIdx += 1;
}
/* ============================================================
   TOURNAMENT
   ============================================================ */
let presetsLoaded = false;

async function loadPresets() {
  if (presetsLoaded) return;
  try {
    const data = await api("/api/tournament/presets");
    presetsLoaded = true;
    $("#presetGrid").innerHTML = data.presets.map((p) => `
      <div class="preset-card" data-preset="${esc(p.id)}" data-format="${p.format}">
        <span class="pbadge">${esc(p.badge || p.format.toUpperCase())}</span>
        <h4>${esc(p.title)}</h4>
        <p>${esc(p.description)}</p>
        <span class="pcount">▸ ${p.team_count} teams · click to simulate</span>
      </div>`).join("");
  } catch (e) {
    $("#presetGrid").innerHTML = `<div class="loading-block">Could not load presets: ${esc(e.message)}</div>`;
  }
}

$("#presetGrid").addEventListener("click", (e) => {
  const card = e.target.closest(".preset-card");
  if (!card) return;
  runTournament({ preset: card.dataset.preset, format: card.dataset.format });
});

$("#tourSims").addEventListener("input", (e) => {
  $("#tourSimLabel").textContent = fmtInt(+e.target.value);
});

const customTeams = [];
const customPicker = createPicker("customPicker", {
  defaultYear: 2022,
  onSelect: (t) => {
    if (customTeams.some((c) => c.name === t.name && c.year === t.year)) {
      toast(`${t.name} (${t.year}) is already in the tournament.`);
      return;
    }
    customTeams.push(t);
    renderCustomChips();
  },
});

function tourneyFormat() {
  const r = document.querySelector('input[name="tformat"]:checked');
  return r ? r.value : "groups";
}

function renderCustomChips() {
  const row = $("#customChips");
  const fmt = tourneyFormat();
  row.innerHTML = customTeams.length
    ? customTeams.map((t, i) => `<span class="team-chip">${esc(t.name)} (${t.year})<button data-i="${i}" title="Remove">✕</button></span>`).join("")
    : `<span class="placeholder">No teams added yet</span>`;
  const badge = $("#customCount");
  badge.textContent = `${customTeams.length} teams`;
  const ok = fmt === "groups"
    ? (customTeams.length >= 8 && customTeams.length % 4 === 0)
    : customTeams.length >= 4;
  badge.classList.toggle("ok", ok);
}

$("#customChips").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-i]");
  if (!btn) return;
  customTeams.splice(+btn.dataset.i, 1);
  renderCustomChips();
});
document.querySelectorAll('input[name="tformat"]').forEach((r) =>
  r.addEventListener("change", renderCustomChips));

$("#runCustom").addEventListener("click", () => {
  const fmt = tourneyFormat();
  if (fmt === "groups" && !(customTeams.length >= 8 && customTeams.length % 4 === 0)) {
    toast("Groups + Knockout needs 8, 12, 16, 20 … teams (multiple of 4).");
    return;
  }
  if (fmt === "knockout" && customTeams.length < 4) {
    toast("Knockout needs at least 4 teams.");
    return;
  }
  runTournament({
    teams: customTeams.map((t) => ({ name: t.name, year: t.year })),
    format: fmt,
  });
});
async function runTournament(payload) {
  const box = $("#tourneyResults");
  box.classList.remove("hidden");
  if (!payload.n_simulations) payload.n_simulations = +$("#tourSims").value;
  const n = payload.n_simulations;
  box.innerHTML = `<div class="progress-note">Simulating tournament — ${fmtInt(n)} full runs of every group match and knockout tie…</div>`;
  box.scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    const res = await api("/api/tournament/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderTournament(res);
  } catch (e) {
    box.innerHTML = `<div class="progress-note">Tournament failed: ${esc(e.message)}</div>`;
  }
}

function podiumSpot(medal, cls, label, pct, sub) {
  return `<div class="podium-spot ${cls}">
    <div class="medal">${medal}</div><h4>${esc(label || "—")}</h4>
    <div class="pct">${pct != null ? Number(pct).toFixed(1) + "%" : "—"}</div>
    <div class="sub">${sub}</div>
  </div>`;
}

function renderTournament(res) {
  const box = $("#tourneyResults");
  const teams = res.teams, meta = res.meta, pod = res.podium, sc = res.showcase;
  const isGroups = meta.format === "groups";
  const maxChamp = Math.max(...teams.map((t) => t.champion_pct), 1);

  const oddsRows = teams.map((t, i) => `
    <div class="odds-row">
      <span class="rank">${i + 1}</span>
      <span class="oname" title="${esc(t.label)} · OVR ${t.overall}">${esc(t.label)}</span>
      <span class="opct">${t.champion_pct.toFixed(1)}%</span>
      <div class="obar"><div data-w="${(t.champion_pct / maxChamp * 100).toFixed(1)}"></div></div>
    </div>`).join("");

  const stageCols = isGroups
    ? ["final_pct", "semi_pct", "qf_pct", "advance_pct", "exp_points"]
    : ["final_pct", "semi_pct", "qf_pct"];
  const stageTitles = { final_pct: "Final", semi_pct: "Semi", qf_pct: "QF", advance_pct: "Advance", exp_points: "xPts" };
  const stageTable = `<table class="chall-table"><thead><tr><th>Team</th>${stageCols.map((c) => `<th>${stageTitles[c]}</th>`).join("")}</tr></thead><tbody>
    ${teams.slice(0, 16).map((t) => `<tr><td>${esc(t.label)}</td>${stageCols.map((c) =>
      `<td>${c === "exp_points" ? t[c] : fmtPct(t[c])}</td>`).join("")}</tr>`).join("")}
  </tbody></table>`;

  const groupsHtml = (isGroups && sc.groups) ? `
    <div class="card"><h3>📑 Showcase Run — Group Stage</h3>
      <div class="groups-grid">${sc.groups.map((g) => {
        return `<div class="group-table"><h5>${esc(g.name)}</h5>
          ${g.standings.map((r, i) => `<div class="grow${i < 2 ? " qualified" : ""}">
            <span>${esc(r.team)}</span><span>${r.w}-${r.d}-${r.l}</span>
            <span>${r.gd > 0 ? "+" : ""}${r.gd}</span><span class="gpts">${r.pts}</span></div>`).join("")}
        </div>`;
      }).join("")}</div>
    </div>` : "";

  const bracketHtml = sc.rounds ? sc.rounds.map((round) => `
    <div class="bracket-round"><h5>${esc(round.name)}</h5>
      ${round.matches.map((m) => {
        const hw = m.winner === m.home, aw = m.winner === m.away;
        return `<div class="bmatch">
          <div class="bt${hw ? " winner" : ""}"><span>${esc(m.home.replace(/ \(\d+\)$/, ""))}</span><span class="score">${m.home_goals}</span></div>
          <div class="bt${aw ? " winner" : ""}"><span>${esc(m.away.replace(/ \(\d+\)$/, ""))}</span><span class="score">${m.away_goals}</span></div>
          ${m.penalties ? `<div class="pens">⚽ Pens ${esc(m.penalties.score)} — ${esc(m.penalties.winner.replace(/ \(\d+\)$/, ""))}</div>` : ""}
        </div>`;
      }).join("")}
    </div>`).join("") : "";

  box.innerHTML = `
  <div class="champ-banner">🏆 ${esc(pod.champion)} — ${pod.champion_pct.toFixed(1)}% to win
    <div style="font-size:0.85rem;font-weight:400;letter-spacing:0.4px;color:var(--muted)">
      ${fmtInt(meta.n_simulations)} tournament runs · ${meta.n_teams} teams · engine ${esc(meta.engine)} · ${meta.duration_ms}ms
    </div>
  </div>

  <div class="podium">
    ${podiumSpot("🥈", "", pod.runner_up, pod.runner_up_pct, "Runner-up (most likely)")}
    ${podiumSpot("🥇", "first", pod.champion, pod.champion_pct, "Champion (most likely)")}
    ${podiumSpot("🥉", "", pod.third, pod.third_pct, "Semi-finalist")}
  </div>

  ${res.most_likely_final ? `<div class="card"><h3>🔮 Most Likely Final (${res.most_likely_final.pct.toFixed(1)}%)</h3>
    <div class="prob-head"><span class="ph-team">${esc(res.most_likely_final.teams[0])}</span>
    <span class="ph-team">${esc(res.most_likely_final.teams[1])}</span></div></div>` : ""}

  <div class="res-grid-2">
    <div class="card"><h3>📈 Championship Odds — all ${teams.length} teams</h3>
      <div style="max-height:520px;overflow-y:auto">${oddsRows}</div>
    </div>
    <div class="card"><h3>🎯 Stage Reach (top 16)</h3>${stageTable}</div>
  </div>

  ${groupsHtml}

  <div class="card"><h3>🌲 Showcase Run — Knockout Bracket (one random simulated run)</h3>
    <div class="bracket">${bracketHtml}</div>
  </div>`;

  animateBars(box);
  box.scrollIntoView({ behavior: "smooth", block: "start" });
}
/* ============================================================
   CHAMPION MODEL — verified metrics dashboard
   ============================================================ */
let championLoaded = false;

async function loadChampion() {
  if (championLoaded) return;
  const box = $("#championContent");
  try {
    const data = await api("/api/champion");
    championLoaded = true;
    const cfg = data.config, ev = cfg.evaluation.metrics, champ = data.results;
    const ens = [
      ["LightGBM", 30.4], ["XGBoost", 25.0], ["HistGBDT", 20.9],
      ["CatBoost", 17.2], ["Dixon-Coles", 6.5],
    ];
    box.innerHTML = `
    <div class="champ-hero">
      <div class="card">
        <h3>🏆 ${esc(cfg.champion_name)} — verified ${esc(cfg.verified_date)}</h3>
        <div class="metric-grid">
          <div class="metric-card hero"><div class="mc-label">Out-of-sample accuracy</div>
            <div class="mc-value">${(ev.accuracy * 100).toFixed(2)}%</div></div>
          <div class="metric-card"><div class="mc-label">Correct</div>
            <div class="mc-value">${fmtInt(cfg.champion_correct)} / ${fmtInt(cfg.test_set_size)}</div></div>
          <div class="metric-card"><div class="mc-label">Log loss</div>
            <div class="mc-value">${ev.log_loss.toFixed(4)}</div></div>
          <div class="metric-card"><div class="mc-label">Normalised RPS</div>
            <div class="mc-value">${ev.normalized_rps.toFixed(4)}</div></div>
          <div class="metric-card"><div class="mc-label">Brier score</div>
            <div class="mc-value">${ev.brier_score.toFixed(4)}</div></div>
          <div class="metric-card"><div class="mc-label">Calibration (ECE)</div>
            <div class="mc-value">${ev.ece.toFixed(4)}</div></div>
        </div>
      </div>
      <div class="card">
        <h3>🧬 Ensemble composition</h3>
        ${ens.map(([n, w]) => `<div class="ens-row"><span class="ename">${n}</span>
          <div class="bar-track"><div class="bar-fill" data-w="${(w / 30.4 * 100).toFixed(1)}"></div></div>
          <span class="epct">~${w}%</span></div>`).join("")}
        <p class="panel-sub" style="margin:8px 0 0">SLSQP convex blend on validation log-loss + temperature scaling · ${cfg.feature_pipeline.n_features} leakage-free features · 49,520 matches · 4 rolling-origin folds</p>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px">
      <h3>📉 Champion vs every challenger (+${(champ.absolute_improvement.accuracy * 100).toFixed(2)}pp over baseline)</h3>
      <table class="chall-table"><thead><tr><th>Model</th><th>Accuracy</th><th>vs Champion</th><th>Why it lost</th></tr></thead><tbody>
        <tr class="champ-row"><td>🏆 Champion Ensemble (R1)</td><td>60.14%</td><td>—</td><td>SLSQP blend of 4 decorrelated GBDTs + Dixon-Coles</td></tr>
        ${data.challengers.map((c) => `<tr><td>${esc(c.name)}</td><td>${c.accuracy.toFixed(2)}%</td>
          <td>${esc(c.delta)}</td><td>${esc(c.note)}</td></tr>`).join("")}
      </tbody></table>
    </div>

    <div class="card">
      <h3>🌍 World Cup 2026 backtest — ${data.wc2026_backtest.accuracy}% on 104 matches</h3>
      <div class="stat-cards">
        <div class="stat-card"><div class="sc-label">Predicted champion (rank #1)</div><div class="sc-value gold">${esc(data.wc2026_backtest.predicted_champion)}</div></div>
        <div class="stat-card"><div class="sc-label">Actual champion</div><div class="sc-value green">${esc(data.wc2026_backtest.actual_champion)}</div></div>
        <div class="stat-card"><div class="sc-label">Tournament accuracy</div><div class="sc-value blue">${data.wc2026_backtest.accuracy}%</div></div>
        <div class="stat-card"><div class="sc-label">Matches evaluated</div><div class="sc-value">${data.wc2026_backtest.matches}</div></div>
      </div>
    </div>`;
    animateBars(box);
  } catch (e) {
    box.innerHTML = `<div class="loading-block">Could not load champion metrics: ${esc(e.message)}</div>`;
  }
}








