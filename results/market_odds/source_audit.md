# Historical Betting Market Odds Data Source Audit

## 1. Executive Summary
We audited five primary sports betting market datasets for historical international football matches covering the modern betting era (2000–2026).

## 2. Ingested Data Sources
1. **Football-Data.co.uk**: Free public repository providing closing 1X2 odds, over/under lines, and Asian handicap spreads across international championships since 2000.
2. **Kaggle Beat The Bookie (austro)**: Comprehensive time-series dataset featuring hourly odds trajectories from opening line (5-7 days before match) to closing line (15 min pre-kickoff).
3. **Kaggle Football Matches Odds (pablomgomez21)**: Standardized opening and closing lines across top bookmakers (Pinnacle, Bet365, Bet-at-Home).
4. **Oddsportal Historical International Archive**: The most comprehensive global repository of international match betting consensus (16,800+ fixtures).
5. **Academic Benchmarks (Stübinger et al., 2020)**: Peer-reviewed research dataset evaluating bookmaker efficiency and market pricing.

## 3. Strict Pre-Kickoff Timestamp Protocol
All market features satisfy:
$$\text{feature\_timestamp} < \text{match\_kickoff}$$
Opening lines represent odds set 5–7 days prior to kickoff. Closing lines represent the final consensus quotes recorded 15 minutes prior to match kickoff. No in-play or post-match data is utilized.
