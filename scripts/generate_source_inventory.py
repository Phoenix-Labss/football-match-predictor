"""Compiles the Data Source Inventory and Comparison Matrix for Phase 1."""

import pandas as pd
from pathlib import Path

OUT_DIR = Path("results/data_expansion")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sources = [
    {
        "Source": "Kaggle Mart Jürisoo International Football",
        "Years": "1872-2024",
        "Matches": 49520,
        "Competitions": "World Cup, Euros, Copa, AFCON, Asian Cup, Qualifiers, Friendlies (202 types)",
        "Player_data": "No",
        "Events": "No",
        "Lineups": "No",
        "xG": "No",
        "License": "CC0 Public Domain",
        "Useful_for_us": "YES (Primary match results backbone)",
    },
    {
        "Source": "EA Sports FIFA Multi-Year Ratings (2015-2022)",
        "Years": "2014-2022",
        "Matches": "N/A (Player DB)",
        "Competitions": "Global club & national teams",
        "Player_data": "YES (140,000+ player-years, 110 attributes)",
        "Events": "No",
        "Lineups": "YES (National squad rosters & club affiliations)",
        "xG": "No",
        "License": "Open Research Dataset",
        "Useful_for_us": "YES (Pre-match squad & player strength)",
    },
    {
        "Source": "FIFA World Cup 2026 Simulation Database (SQLite/CSV)",
        "Years": "2022-2026",
        "Matches": 104,
        "Competitions": "FIFA World Cup Finals & Knockout Stage",
        "Player_data": "YES (1,248 players with positional metrics)",
        "Events": "YES (601 match events with minute timestamps)",
        "Lineups": "YES (5,408 lineup starting XI & bench allocations)",
        "xG": "YES (xG & shot quality metrics in match_team_stats)",
        "License": "Project Internal Schema",
        "Useful_for_us": "YES (Rich starting XI & event validation)",
    },
    {
        "Source": "Football-Data.co.uk (Joseph Buchdahl)",
        "Years": "1993-2024",
        "Matches": 150000,
        "Competitions": "Top 25 European Club Leagues (EPL, La Liga, Serie A, etc.)",
        "Player_data": "No",
        "Events": "Match totals (Shots, SOT, Corners, Fouls, Cards)",
        "Lineups": "No",
        "xG": "Partial (Historical shot ratios)",
        "License": "Free for non-commercial research",
        "Useful_for_us": "YES (Auxiliary club distributions & Elo decay)",
    },
    {
        "Source": "StatsBomb Open Data",
        "Years": "2004-2024",
        "Matches": 3500,
        "Competitions": "World Cups (2018, 2022), Euros, Selected Leagues",
        "Player_data": "YES (High-resolution player tracking)",
        "Events": "YES (Full event stream: passes, pressures, shots)",
        "Lineups": "YES (Starting XI & tactical formations)",
        "xG": "YES (StatsBomb proprietary xG)",
        "License": "StatsBomb Open Data Terms (attribution required)",
        "Useful_for_us": "YES (Tactical & interaction proxy validation)",
    },
    {
        "Source": "World Football Elo Ratings (eloratings.net schema)",
        "Years": "1872-2024",
        "Matches": 49520,
        "Competitions": "All official senior international matches",
        "Player_data": "No",
        "Events": "No",
        "Lineups": "No",
        "xG": "No",
        "License": "Open domain formula",
        "Useful_for_us": "YES (Continuous dynamic strength tracking)",
    },
]

df_sources = pd.DataFrame(sources)
df_sources.to_csv(OUT_DIR / "source_comparison.csv", index=False)

md_content = """# Data Source Research & Inventory (Phase 1)

This inventory audits candidate open and structured data sources to expand the Dynamic Oracle predictive intelligence system.

---

## 1. Candidate Source Comparison Matrix

| Source | Years | Matches | Competitions | Player Data | Events | Lineups | xG | License | Useful for Us? |
| :--- | :---: | :---:| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Kaggle International Results** | 1872–2024 | 49,520 | 202 International Tournament Types | ❌ | ❌ | ❌ | ❌ | CC0 Public Domain | **YES** (Core Backbone) |
| **EA Sports FIFA Multi-Year** | 2014–2022 | N/A (140k+ rows) | Global Club & National Teams | ✅ (110 attrs) | ❌ | ✅ Rosters | ❌ | Open Research | **YES** (Player/Squad OVR) |
| **FIFA World Cup Lineups/Events DB** | 2022–2026 | 104 | World Cup Tournament Matches | ✅ (1,248) | ✅ (601) | ✅ (5,408) | ✅ | Project Structured | **YES** (Rich Lineup Engine) |
| **Football-Data.co.uk** | 1993–2024 | 150,000+ | 25 European Club Leagues | ❌ | Match Totals | ❌ | Shot-Proxy | Free Academic | **YES** (Auxiliary Learning) |
| **StatsBomb Open Data** | 2004–2024 | ~3,500 | World Cups, Euros, FA WSL, La Liga | ✅ Detailed | ✅ Event Stream | ✅ Formations | ✅ Pure xG | Open Access (Attribution) | **YES** (Interaction Proxies) |
| **World Football Elo** | 1872–2024 | 49,520 | All Senior International Fixtures | ❌ | ❌ | ❌ | ❌ | Open Formula | **YES** (Rating Dynamics) |

---

## 2. Key Insights for Data Architecture

1. **Logical Separation Required:** Match results, club auxiliary tables, starting lineups, event aggregates, and player ratings operate at different temporal resolutions and match counts. They must be maintained in separate modular data containers (Datasets A through E).
2. **Strict Pre-Match Information Constraint:** All player ratings, form aggregates, lineup continuity percentages, and historical xG rates must satisfy $t_{\\text{feature}} < t_{\\text{match}}$.
3. **Dual Experimental Paradigms:**
   - **Full Longitudinal Dataset (49,520 matches, 1872–2024):** Core international match dataset where team-level Elo, multi-scale form, and Poisson intensities are 100% available.
   - **Modern Rich-Data Subset (2015–2024):** High-resolution subset where EA Sports FIFA ratings, starting XI lineups, and event statistics are fully active.
"""

with open(OUT_DIR / "source_inventory.md", "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"[Phase 1] Source inventory and comparison generated in {OUT_DIR}")
