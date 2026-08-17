/**
 * Dynamic Oracle — Football Match Outcome Predictor & Simulator
 * Front-end Logic & Interactive Match Engine Controller
 */

const API_BASE = ""; // relative paths for same-origin API

// Country flag & club crest emoji mapping
const TEAM_EMOJIS = {
  "FC Barcelona": "🔵🔴",
  "Real Madrid": "👑",
  "Manchester City": "🔵",
  "Liverpool": "🔴",
  "FC Bayern München": "🔴⚪",
  "Paris Saint-Germain": "🗼",
  "Juventus": "⚪⚫",
  "Chelsea": "🦁",
  "Arsenal": "🔴⚪",
  "Manchester United": "👹",
  "Morocco": "🇲🇦",
  "Argentina": "🇦🇷",
  "France": "🇫🇷",
  "Brazil": "🇧🇷",
  "Germany": "🇩🇪",
  "Spain": "🇪🇸",
  "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
  "Portugal": "🇵🇹",
  "Netherlands": "🇳🇱",
  "Italy": "🇮🇹",
  "Croatia": "🇭🇷",
  "Belgium": "🇧🇪",
  "Uruguay": "🇺🇾",
  "Japan": "🇯🇵",
  "United States": "🇺🇸",
  "Mexico": "🇲🇽",
  "Senegal": "🇸🇳",
  "Korea Republic": "🇰🇷",
};

function getTeamBadge(name) {
  if (TEAM_EMOJIS[name]) return TEAM_EMOJIS[name];
  if (name.toLowerCase().includes("barcelona")) return "🔵🔴";
  if (name.toLowerCase().includes("madrid")) return "👑";
  if (name.toLowerCase().includes("manchester")) return "⚽";
  if (name.toLowerCase().includes("bayern")) return "🔴⚪";
  return "⚽";
}

// Application State
const state = {
  teamA: {
    name: "FC Barcelona",
    year: 2015,
    type: "all",
    formation: "4-3-3",
  },
  teamB: {
    name: "Morocco",
    year: 2026,
    type: "all",
    formation: "4-3-3",
  },
  activeLineupTab: "a",
  lastResult: null,
};

// Debounce helper
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// --------------------------------------------------------------------------
// INITIALIZATION
// --------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  initApp();
});

async function initApp() {
  setupEventListeners();
  await loadMetadata();
  await updateTeamPreview("a");
  await updateTeamPreview("b");
  
  // Auto-simulate default matchup (FC Barcelona 2015 vs Morocco 2026)
  runSimulation();
}

async function loadMetadata() {
  try {
    const res = await fetch(`${API_BASE}/api/meta`);
    const data = await res.json();
    
    // Set status
    const statusBadge = document.getElementById("system-status");
    if (data.status === "online") {
      statusBadge.classList.add("online");
      statusBadge.querySelector(".status-label").textContent = "ML Engine Live";
    }
  } catch (err) {
    console.warn("Could not load /api/meta:", err);
  }
}

// --------------------------------------------------------------------------
// EVENT LISTENERS SETUP
// --------------------------------------------------------------------------
function setupEventListeners() {
  // Simulate button
  document.getElementById("btn-simulate").addEventListener("click", runSimulation);

  // Year selectors
  document.getElementById("year-a").addEventListener("change", (e) => {
    state.teamA.year = parseInt(e.target.value);
    updateTeamPreview("a");
  });
  document.getElementById("year-b").addEventListener("change", (e) => {
    state.teamB.year = parseInt(e.target.value);
    updateTeamPreview("b");
  });

  // Type pills A
  document.querySelectorAll("#type-pills-a .type-pill").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#type-pills-a .type-pill").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.teamA.type = btn.dataset.type;
      fetchTeamsForDropdown("a", document.getElementById("search-a").value);
    });
  });

  // Type pills B
  document.querySelectorAll("#type-pills-b .type-pill").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#type-pills-b .type-pill").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.teamB.type = btn.dataset.type;
      fetchTeamsForDropdown("b", document.getElementById("search-b").value);
    });
  });

  // Autocomplete search inputs
  const searchA = document.getElementById("search-a");
  const searchB = document.getElementById("search-b");

  searchA.addEventListener("focus", () => fetchTeamsForDropdown("a", searchA.value));
  searchA.addEventListener("input", debounce(() => fetchTeamsForDropdown("a", searchA.value), 200));

  searchB.addEventListener("focus", () => fetchTeamsForDropdown("b", searchB.value));
  searchB.addEventListener("input", debounce(() => fetchTeamsForDropdown("b", searchB.value), 200));

  // Hide dropdowns on click outside
  document.addEventListener("click", (e) => {
    if (!e.target.closest("#panel-team-a")) {
      document.getElementById("dropdown-a").classList.add("hidden");
    }
    if (!e.target.closest("#panel-team-b")) {
      document.getElementById("dropdown-b").classList.add("hidden");
    }
  });

  // Preset Selector
  document.getElementById("preset-select").addEventListener("change", handlePresetChange);

  // Lineup tabs
  document.getElementById("tab-lineup-a").addEventListener("click", () => switchLineupTab("a"));
  document.getElementById("tab-lineup-b").addEventListener("click", () => switchLineupTab("b"));
}

// --------------------------------------------------------------------------
// PRESETS HANDLER
// --------------------------------------------------------------------------
function handlePresetChange(e) {
  const presetId = e.target.value;
  const presets = {
    barca15_morocco26: { teamA: "FC Barcelona", yearA: 2015, teamB: "Morocco", yearB: 2026 },
    real17_mancity22: { teamA: "Real Madrid", yearA: 2017, teamB: "Manchester City", yearB: 2022 },
    argentina22_france22: { teamA: "Argentina", yearA: 2022, teamB: "France", yearB: 2022 },
    bayern20_liverpool19: { teamA: "FC Bayern München", yearA: 2020, teamB: "Liverpool", yearB: 2019 },
    brazil15_germany15: { teamA: "Brazil", yearA: 2015, teamB: "Germany", yearB: 2015 },
  };

  if (presets[presetId]) {
    const p = presets[presetId];
    state.teamA.name = p.teamA;
    state.teamA.year = p.yearA;
    state.teamB.name = p.teamB;
    state.teamB.year = p.yearB;

    document.getElementById("year-a").value = p.yearA;
    document.getElementById("search-a").value = p.teamA;
    document.getElementById("year-b").value = p.yearB;
    document.getElementById("search-b").value = p.teamB;

    updateTeamPreview("a");
    updateTeamPreview("b");
    runSimulation();
  }
}

// --------------------------------------------------------------------------
// TEAM SEARCH & AUTOCOMPLETE DROPDOWN
// --------------------------------------------------------------------------
async function fetchTeamsForDropdown(side, query) {
  const year = side === "a" ? state.teamA.year : state.teamB.year;
  const type = side === "a" ? state.teamA.type : state.teamB.type;
  const dropdown = document.getElementById(`dropdown-${side}`);

  try {
    const res = await fetch(`${API_BASE}/api/teams?year=${year}&type=${type}&q=${encodeURIComponent(query)}&limit=30`);
    const data = await res.json();

    if (!data.teams || data.teams.length === 0) {
      dropdown.innerHTML = `<div class="dropdown-item" style="color: var(--text-muted);">No teams found for ${year}</div>`;
      dropdown.classList.remove("hidden");
      return;
    }

    dropdown.innerHTML = data.teams.map(t => `
      <div class="dropdown-item" data-team="${t.name}">
        <div>
          <strong>${t.name}</strong>
          <span style="color: var(--text-muted); font-size: 0.75rem; margin-left: 6px;">★ ${t.avg_rating}</span>
        </div>
        <span class="team-type-badge">${t.type === "national" ? "Nation" : "Club"}</span>
      </div>
    `).join("");

    dropdown.classList.remove("hidden");

    dropdown.querySelectorAll(".dropdown-item").forEach(item => {
      item.addEventListener("click", () => {
        const selectedTeam = item.dataset.team;
        if (side === "a") {
          state.teamA.name = selectedTeam;
          document.getElementById("search-a").value = selectedTeam;
          updateTeamPreview("a");
        } else {
          state.teamB.name = selectedTeam;
          document.getElementById("search-b").value = selectedTeam;
          updateTeamPreview("b");
        }
        dropdown.classList.add("hidden");
      });
    });
  } catch (err) {
    console.error(`Error fetching teams for side ${side}:`, err);
  }
}

// --------------------------------------------------------------------------
// TEAM QUICK PREVIEWS
// --------------------------------------------------------------------------
async function updateTeamPreview(side) {
  const teamName = side === "a" ? state.teamA.name : state.teamB.name;
  const year = side === "a" ? state.teamA.year : state.teamB.year;

  document.getElementById(`display-name-${side}`).textContent = teamName;
  document.getElementById(`badge-${side}`).textContent = getTeamBadge(teamName);

  try {
    const res = await fetch(`${API_BASE}/api/team-preview?team=${encodeURIComponent(teamName)}&year=${year}`);
    if (!res.ok) throw new Error("Preview not found");
    const data = await res.json();

    document.getElementById(`prev-ovr-${side}`).textContent = data.overall;
    document.getElementById(`prev-att-${side}`).textContent = data.attack;
    document.getElementById(`prev-mid-${side}`).textContent = data.midfield;
    document.getElementById(`prev-def-${side}`).textContent = data.defence;

    const starsList = document.getElementById(`stars-list-${side}`);
    if (data.top_players && data.top_players.length > 0) {
      starsList.innerHTML = data.top_players.slice(0, 3).map(p => `
        <span class="player-tag">${p.name} (${p.overall})</span>
      `).join("");
    }
  } catch (err) {
    console.warn(`Could not load preview for ${teamName} (${year})`);
  }
}

// --------------------------------------------------------------------------
// RUN SIMULATION
// --------------------------------------------------------------------------
async function runSimulation() {
  const teamA = state.teamA.name;
  const yearA = state.teamA.year;
  const teamB = state.teamB.name;
  const yearB = state.teamB.year;
  const simCount = parseInt(document.getElementById("sim-count").value) || 10000;
  const venueMode = document.getElementById("venue-mode").value;
  const neutral = venueMode === "neutral";

  const btnSimulate = document.getElementById("btn-simulate");
  const loadingState = document.getElementById("loading-state");
  const errorBanner = document.getElementById("error-banner");
  const resultsSection = document.getElementById("results-section");

  document.getElementById("loading-sim-count").textContent = simCount.toLocaleString();

  btnSimulate.disabled = true;
  loadingState.classList.remove("hidden");
  errorBanner.classList.add("hidden");
  resultsSection.classList.add("hidden");

  const payload = {
    team_a: teamA,
    year_a: yearA,
    team_b: teamB,
    year_b: yearB,
    n_simulations: simCount,
    neutral: neutral,
    formation_a: state.teamA.formation,
    formation_b: state.teamB.formation,
  };

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Simulation failed.");
    }

    const data = await res.json();
    state.lastResult = data;
    renderResults(data);

    resultsSection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    console.error("Simulation error:", err);
    errorBanner.querySelector("#error-message").textContent = err.message;
    errorBanner.classList.remove("hidden");
  } finally {
    loadingState.classList.add("hidden");
    btnSimulate.disabled = false;
  }
}

// --------------------------------------------------------------------------
// RENDER RESULTS
// --------------------------------------------------------------------------
function renderResults(data) {
  // 1. HERO SCOREBOARD
  document.getElementById("res-badge-a").textContent = getTeamBadge(data.team_a.name);
  document.getElementById("res-badge-b").textContent = getTeamBadge(data.team_b.name);
  document.getElementById("res-name-a").textContent = data.team_a.name;
  document.getElementById("res-name-b").textContent = data.team_b.name;
  document.getElementById("res-year-a").textContent = `${data.team_a.year_resolved} Edition`;
  document.getElementById("res-year-b").textContent = `${data.team_b.year_resolved} Edition`;
  
  document.getElementById("res-scoreline").textContent = data.most_likely_scoreline;
  document.getElementById("res-confidence-badge").querySelector(".conf-text").textContent = 
    `${data.confidence_pct}% Model Confidence`;

  // 2. PROBABILITY METER
  document.getElementById("lbl-prob-a").textContent = data.team_a.name;
  document.getElementById("lbl-prob-b").textContent = data.team_b.name;
  document.getElementById("val-prob-a").textContent = `${data.probabilities.team_a_win_pct}%`;
  document.getElementById("val-prob-draw").textContent = `${data.probabilities.draw_pct}%`;
  document.getElementById("val-prob-b").textContent = `${data.probabilities.team_b_win_pct}%`;

  document.getElementById("bar-prob-a").style.width = `${data.probabilities.team_a_win_pct}%`;
  document.getElementById("bar-prob-draw").style.width = `${data.probabilities.draw_pct}%`;
  document.getElementById("bar-prob-b").style.width = `${data.probabilities.team_b_win_pct}%`;

  // 3. EXPECTED GOALS (xG) & EXTRA MARKETS
  document.getElementById("val-xg-a").textContent = data.xg.team_a;
  document.getElementById("val-xg-b").textContent = data.xg.team_b;
  document.getElementById("val-xg-total").textContent = data.xg.total;

  if (data.extra_markets) {
    document.getElementById("val-over25").textContent = `${data.extra_markets.over_2_5_goals_pct}%`;
    document.getElementById("val-btts").textContent = `${data.extra_markets.btts_pct}%`;
    
    if (data.extra_markets.knockout_penalties) {
      document.getElementById("lbl-pen-a").textContent = `${data.team_a.name} in PKs:`;
      document.getElementById("lbl-pen-b").textContent = `${data.team_b.name} in PKs:`;
      document.getElementById("val-pen-a").textContent = `${data.extra_markets.knockout_penalties.team_a_pen_win_pct}%`;
      document.getElementById("val-pen-b").textContent = `${data.extra_markets.knockout_penalties.team_b_pen_win_pct}%`;
    }
  }

  // 4. TOP PROBABLE SCORELINES
  const slMatrix = document.getElementById("scorelines-matrix");
  slMatrix.innerHTML = data.top_scorelines.map((sl, idx) => `
    <div class="scoreline-pill ${idx === 0 ? 'top-pick' : ''}">
      <div class="sl-score">${sl.scoreline}</div>
      <div class="sl-pct">${sl.pct}%</div>
      <div class="sl-count">${sl.count.toLocaleString()} sims</div>
    </div>
  `).join("");

  // 5. TACTICAL & ATTRIBUTE CHANNELS
  renderAttributeChannels(data.team_a, data.team_b);

  // 6. TIMELINE HIGHLIGHTS
  renderTimeline(data.timeline);

  // 7. LINEUPS
  renderLineupTab();

  // 8. ML EVALUATION SUITE
  renderEvaluationSuite(data.evaluation_metrics);
}

// --------------------------------------------------------------------------
// ATTRIBUTE CHANNELS RENDERER
// --------------------------------------------------------------------------
function renderAttributeChannels(teamA, teamB) {
  const channels = [
    { label: "Overall Quality", valA: teamA.rating.overall, valB: teamB.rating.overall },
    { label: "Attack Power", valA: teamA.rating.attack, valB: teamB.rating.attack },
    { label: "Midfield Control", valA: teamA.rating.midfield, valB: teamB.rating.midfield },
    { label: "Defensive Solidity", valA: teamA.rating.defence, valB: teamB.rating.defence },
    { label: "Goalkeeping", valA: teamA.rating.gk, valB: teamB.rating.gk },
    { label: "Physicality", valA: teamA.rating.physicality, valB: teamB.rating.physicality },
    { label: "Pace / Speed", valA: teamA.rating.pace, valB: teamB.rating.pace },
  ];

  const listContainer = document.getElementById("channel-bars-list");
  listContainer.innerHTML = channels.map(c => {
    const total = c.valA + c.valB;
    const pctA = ((c.valA / total) * 100).toFixed(1);
    const pctB = (100 - pctA).toFixed(1);

    return `
      <div class="channel-row">
        <div class="channel-meta">
          <span style="color: var(--color-gold); font-family: var(--font-mono);">${c.valA}</span>
          <span style="color: var(--text-secondary);">${c.label}</span>
          <span style="color: var(--color-cyan); font-family: var(--font-mono);">${c.valB}</span>
        </div>
        <div class="channel-track">
          <div class="channel-bar-a" style="width: ${pctA}%;"></div>
          <div class="channel-bar-b" style="width: ${pctB}%;"></div>
        </div>
      </div>
    `;
  }).join("");
}

// --------------------------------------------------------------------------
// TIMELINE RENDERER
// --------------------------------------------------------------------------
function renderTimeline(timeline) {
  const feed = document.getElementById("timeline-feed");
  if (!timeline || timeline.length === 0) {
    feed.innerHTML = `<div style="color: var(--text-muted); font-size: 0.85rem;">No timeline events generated.</div>`;
    return;
  }

  feed.innerHTML = timeline.map(ev => `
    <div class="timeline-event ${ev.type}">
      <span class="event-min">${ev.minute}'</span>
      <span class="event-desc">${ev.description}</span>
      <span class="event-score">${ev.score || ""}</span>
    </div>
  `).join("");
}

// --------------------------------------------------------------------------
// LINEUP TAB RENDERER
// --------------------------------------------------------------------------
function switchLineupTab(tab) {
  state.activeLineupTab = tab;
  document.getElementById("tab-lineup-a").classList.toggle("active", tab === "a");
  document.getElementById("tab-lineup-b").classList.toggle("active", tab === "b");
  renderLineupTab();
}

function renderLineupTab() {
  if (!state.lastResult) return;
  const team = state.activeLineupTab === "a" ? state.lastResult.team_a : state.lastResult.team_b;
  const container = document.getElementById("lineup-container");

  document.getElementById("tab-lineup-a").textContent = `${state.lastResult.team_a.name} (${state.lastResult.team_a.formation})`;
  document.getElementById("tab-lineup-b").textContent = `${state.lastResult.team_b.name} (${state.lastResult.team_b.formation})`;

  if (!team.lineup || team.lineup.length === 0) {
    container.innerHTML = `<div style="color: var(--text-muted);">Lineup data unavailable.</div>`;
    return;
  }

  container.innerHTML = team.lineup.map(p => `
    <div class="player-card">
      <div class="player-info-main">
        <span class="player-pos-badge">${p.slot_label || p.position} • ${p.slot_group}</span>
        <span class="player-name-text">${p.name}</span>
        <span class="player-club-sub">${p.club} • Age ${p.age}</span>
      </div>
      <div class="player-rating-box">
        <div class="player-ovr-number">${p.overall}</div>
        <div style="font-size: 0.65rem; color: var(--text-muted); font-family: var(--font-mono);">
          PAC ${p.pace} | SHO ${p.shooting}
        </div>
      </div>
    </div>
  `).join("");
}

// --------------------------------------------------------------------------
// ML EVALUATION SUITE RENDERER
// --------------------------------------------------------------------------
function renderEvaluationSuite(metrics) {
  if (!metrics) return;

  document.getElementById("metric-rps").textContent = metrics.headline_rps;
  document.getElementById("metric-logloss").textContent = metrics.log_loss;
  document.getElementById("metric-brier").textContent = metrics.brier_score;
  document.getElementById("metric-ece").textContent = metrics.ece_calibration_error;

  const calibContainer = document.getElementById("calib-bars");
  if (metrics.calibration_curve && metrics.calibration_curve.length > 0) {
    calibContainer.innerHTML = metrics.calibration_curve.map(bin => `
      <div class="calib-row">
        <span class="calib-bin-lbl">${bin.bin}</span>
        <div class="calib-bar-track">
          <div class="calib-fill" style="width: ${bin.acc * 100}%;"></div>
        </div>
        <span class="calib-acc-lbl">${(bin.acc * 100).toFixed(0)}%</span>
      </div>
    `).join("");
  }
}
