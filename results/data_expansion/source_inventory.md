# Data Source Research & Inventory (Phase 1)

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
2. **Strict Pre-Match Information Constraint:** All player ratings, form aggregates, lineup continuity percentages, and historical xG rates must satisfy $t_{\text{feature}} < t_{\text{match}}$.
3. **Dual Experimental Paradigms:**
   - **Full Longitudinal Dataset (49,520 matches, 1872–2024):** Core international match dataset where team-level Elo, multi-scale form, and Poisson intensities are 100% available.
   - **Modern Rich-Data Subset (2015–2024):** High-resolution subset where EA Sports FIFA ratings, starting XI lineups, and event statistics are fully active.
