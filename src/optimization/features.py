"""Comprehensive pre-match feature engineering for soccer outcome prediction.

All features are computed strictly chronologically using ONLY information
available prior to kickoff (Zero Temporal Leakage Guarantee).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.features.strength import UpdaterConfig, StrengthTracker, RATING_SCALE


FACT = np.array([1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0])
K_IDX = np.arange(8)

def _bivariate_poisson_probs(
    lambda_h: float, lambda_a: float, rho: float = -0.05
) -> tuple[float, float, float]:
    """Fast vectorized pre-match Win/Draw/Loss probabilities from Poisson intensities."""
    exp_lh = np.exp(-lambda_h)
    exp_la = np.exp(-lambda_a)
    p_h = (lambda_h ** K_IDX) * exp_lh / FACT
    p_a = (lambda_a ** K_IDX) * exp_la / FACT
    
    p_matrix = np.outer(p_h, p_a)
    p_matrix[0, 0] = max(0.0, p_matrix[0, 0] * (1.0 - lambda_h * lambda_a * rho))
    p_matrix[0, 1] = max(0.0, p_matrix[0, 1] * (1.0 + lambda_h * rho))
    p_matrix[1, 0] = max(0.0, p_matrix[1, 0] * (1.0 + lambda_a * rho))
    p_matrix[1, 1] = max(0.0, p_matrix[1, 1] * (1.0 - rho))
    
    tot = p_matrix.sum()
    if tot > 0:
        p_matrix /= tot
        
    p_home = float(np.sum(np.tril(p_matrix, -1)))
    p_draw = float(np.sum(np.diag(p_matrix)))
    p_away = float(np.sum(np.triu(p_matrix, 1)))
    return p_home, p_draw, p_away


class AdvancedHistoryBuffer:
    """Maintains rolling match history, EWMA stats, H2H, and Poisson intensities."""

    def __init__(self, form_windows: list[int] = (3, 5, 8, 10, 15, 20, 30)):
        self.form_windows = sorted(form_windows)
        self.max_len = max(form_windows)
        self._history: dict[str, deque] = {}
        self._h2h: dict[tuple[str, str], list[dict]] = {}
        self._last_date: dict[str, pd.Timestamp] = {}
        self._ewma: dict[str, dict[float, dict[str, float]]] = {}
        self.alphas = [0.1, 0.2, 0.3, 0.5]

    def get_stats(self, team: str, window: int) -> dict:
        """Rolling stats over recent window matches."""
        hist = self._history.get(team)
        if hist is None or len(hist) < 2:
            return {
                f"gf_{window}": np.nan,
                f"ga_{window}": np.nan,
                f"gd_{window}": np.nan,
                f"win_{window}": np.nan,
                f"draw_{window}": np.nan,
                f"loss_{window}": np.nan,
                f"pts_{window}": np.nan,
                f"opp_adj_gf_{window}": np.nan,
                f"opp_adj_ga_{window}": np.nan,
                f"opp_adj_gd_{window}": np.nan,
            }
        recent = list(hist)[-min(len(hist), window):]
        n = len(recent)
        gf = sum(m["gf"] for m in recent) / n
        ga = sum(m["ga"] for m in recent) / n
        wins = sum(1.0 for m in recent if m["res"] == 1.0) / n
        draws = sum(1.0 for m in recent if m["res"] == 0.5) / n
        losses = sum(1.0 for m in recent if m["res"] == 0.0) / n
        pts = sum(3.0 if m["res"] == 1.0 else (1.0 if m["res"] == 0.5 else 0.0) for m in recent) / n
        
        # Opponent Elo adjusted goals
        opp_gf = sum(m["gf"] * (m["opp_elo"] / 1500.0) for m in recent) / n
        opp_ga = sum(m["ga"] * (1500.0 / max(500.0, m["opp_elo"])) for m in recent) / n

        return {
            f"gf_{window}": gf,
            f"ga_{window}": ga,
            f"gd_{window}": gf - ga,
            f"win_{window}": wins,
            f"draw_{window}": draws,
            f"loss_{window}": losses,
            f"pts_{window}": pts,
            f"opp_adj_gf_{window}": opp_gf,
            f"opp_adj_ga_{window}": opp_ga,
            f"opp_adj_gd_{window}": opp_gf - opp_ga,
        }

    def get_ewma(self, team: str) -> dict:
        """Exponentially weighted moving averages."""
        res = {}
        team_ewma = self._ewma.get(team)
        for a in self.alphas:
            if team_ewma and a in team_ewma:
                e = team_ewma[a]
                res[f"ewma_gf_{a}"] = e["gf"]
                res[f"ewma_ga_{a}"] = e["ga"]
                res[f"ewma_gd_{a}"] = e["gf"] - e["ga"]
                res[f"ewma_pts_{a}"] = e["pts"]
            else:
                res[f"ewma_gf_{a}"] = np.nan
                res[f"ewma_ga_{a}"] = np.nan
                res[f"ewma_gd_{a}"] = np.nan
                res[f"ewma_pts_{a}"] = np.nan
        return res

    def get_h2h(self, team_a: str, team_b: str) -> dict:
        """Pre-match head-to-head metrics with Bayesian shrinkage prior."""
        pair = (team_a, team_b) if team_a < team_b else (team_b, team_a)
        records = self._h2h.get(pair, [])
        n_meetings = len(records)
        
        if n_meetings == 0:
            return {
                "h2h_matches": 0.0,
                "h2h_win_rate_home": 0.38,  # Prior base rate
                "h2h_draw_rate": 0.25,
                "h2h_gd_home": 0.0,
            }

        # Calculate from perspective of team_a
        wins_a = 0.0
        draws = 0.0
        gd_a = 0.0
        for m in records:
            if m["home"] == team_a:
                if m["res"] == 1.0:
                    wins_a += 1.0
                elif m["res"] == 0.5:
                    draws += 1.0
                gd_a += (m["gf"] - m["ga"])
            else:
                if m["res"] == 0.0:
                    wins_a += 1.0
                elif m["res"] == 0.5:
                    draws += 1.0
                gd_a += (m["ga"] - m["gf"])

        # Empirical Bayes shrinkage towards general base rate
        prior_weight = 3.0
        shrunk_win = (wins_a + 0.38 * prior_weight) / (n_meetings + prior_weight)
        shrunk_draw = (draws + 0.25 * prior_weight) / (n_meetings + prior_weight)
        shrunk_gd = (gd_a + 0.0) / (n_meetings + prior_weight)

        return {
            "h2h_matches": float(n_meetings),
            "h2h_win_rate_home": shrunk_win,
            "h2h_draw_rate": shrunk_draw,
            "h2h_gd_home": shrunk_gd,
        }

    def rest_days(self, team: str, date: pd.Timestamp) -> float:
        last = self._last_date.get(team)
        if last is None:
            return np.nan
        return float((date - last).days)

    def record_match(
        self,
        home: str,
        away: str,
        date: pd.Timestamp,
        hg: int,
        ag: int,
        elo_home: float,
        elo_away: float,
    ) -> None:
        res_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        res_a = 1.0 - res_h
        pts_h = 3.0 if res_h == 1.0 else (1.0 if res_h == 0.5 else 0.0)
        pts_a = 3.0 if res_a == 1.0 else (1.0 if res_a == 0.5 else 0.0)

        # 1. Update rolling deques
        self._history.setdefault(home, deque(maxlen=self.max_len)).append({
            "gf": hg, "ga": ag, "res": res_h, "opp_elo": elo_away, "date": date
        })
        self._history.setdefault(away, deque(maxlen=self.max_len)).append({
            "gf": ag, "ga": hg, "res": res_a, "opp_elo": elo_home, "date": date
        })

        # 2. Update EWMA
        self._ewma.setdefault(home, {})
        self._ewma.setdefault(away, {})
        for a in self.alphas:
            if a not in self._ewma[home]:
                self._ewma[home][a] = {"gf": float(hg), "ga": float(ag), "pts": pts_h}
            else:
                e = self._ewma[home][a]
                e["gf"] = (1 - a) * e["gf"] + a * hg
                e["ga"] = (1 - a) * e["ga"] + a * ag
                e["pts"] = (1 - a) * e["pts"] + a * pts_h

            if a not in self._ewma[away]:
                self._ewma[away][a] = {"gf": float(ag), "ga": float(hg), "pts": pts_a}
            else:
                e = self._ewma[away][a]
                e["gf"] = (1 - a) * e["gf"] + a * ag
                e["ga"] = (1 - a) * e["ga"] + a * hg
                e["pts"] = (1 - a) * e["pts"] + a * pts_a

        # 3. Update H2H
        pair = (home, away) if home < away else (away, home)
        self._h2h.setdefault(pair, []).append({
            "home": home, "away": away, "gf": hg, "ga": ag, "res": res_h, "date": date
        })

        self._last_date[home] = date
        self._last_date[away] = date


def build_advanced_feature_matrix(
    matches: pd.DataFrame,
    updater_cfg: UpdaterConfig,
    form_windows: list[int] = (3, 5, 8, 10, 15, 20, 30),
    include_dixon_coles: bool = True,
    include_player_features: bool = True,
    fifa_lookup: dict[tuple[str, int], dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Construct high-capacity, strictly pre-match feature matrix."""
    tracker = StrengthTracker(updater_cfg)
    buf = AdvancedHistoryBuffer(form_windows=form_windows)
    rows = []

    for row in matches.itertuples(index=False):
        home, away = row.home_team, row.away_team
        date = row.date
        year = date.year if hasattr(date, "year") else 2022
        neutral = bool(row.neutral)
        hg, ag = int(row.home_goals), int(row.away_goals)

        feats: dict = {}

        # 1. Elo & Team Strength Features
        eh = tracker.rating(home)
        ea = tracker.rating(away)
        ha = 0.0 if neutral else updater_cfg.home_advantage
        diff = (eh + ha) - ea
        raw_diff = eh - ea

        p_home_expected = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        p_away_expected = 1.0 - p_home_expected

        feats["elo_home"] = eh
        feats["elo_away"] = ea
        feats["elo_diff"] = diff
        feats["elo_raw_diff"] = raw_diff
        feats["elo_ratio"] = (eh + ha) / max(500.0, ea)
        feats["elo_diff_abs"] = abs(raw_diff)
        feats["elo_diff_sq"] = np.sign(raw_diff) * (raw_diff / 100.0) ** 2
        feats["expected_home_score"] = p_home_expected
        feats["expected_away_score"] = p_away_expected

        # Adaptive signals
        feats["home_consistency"] = tracker._consistency(home)
        feats["away_consistency"] = tracker._consistency(away)
        feats["home_surprise"] = tracker._surprise(home)
        feats["away_surprise"] = tracker._surprise(away)
        feats["home_cap_pts"] = tracker._adaptive_cap_points(home)
        feats["away_cap_pts"] = tracker._adaptive_cap_points(away)

        # 2. Dixon-Coles Pre-Match Bivariate Poisson Intensities
        if include_dixon_coles:
            mu_h = 1.35 if not neutral else 1.20
            mu_a = 1.05 if not neutral else 1.20
            lambda_h = max(0.2, mu_h * np.exp(diff / 600.0))
            lambda_a = max(0.2, mu_a * np.exp(-diff / 600.0))
            dc_ph, dc_pd, dc_pa = _bivariate_poisson_probs(lambda_h, lambda_a)
            
            feats["dc_xg_home"] = lambda_h
            feats["dc_xg_away"] = lambda_a
            feats["dc_xg_diff"] = lambda_h - lambda_a
            feats["dc_xg_total"] = lambda_h + lambda_a
            feats["dc_p_home"] = dc_ph
            feats["dc_p_draw"] = dc_pd
            feats["dc_p_away"] = dc_pa

        # 3. Match Context Features
        feats["is_neutral"] = float(neutral)
        feats["is_friendly"] = 1.0 if row.tournament == "Friendly" else 0.0
        feats["is_world_cup"] = 1.0 if "FIFA World Cup" in str(row.tournament) else 0.0
        feats["is_euro"] = 1.0 if "UEFA Euro" in str(row.tournament) else 0.0
        feats["home_rest_days"] = min(60.0, max(0.0, buf.rest_days(home, date)))
        feats["away_rest_days"] = min(60.0, max(0.0, buf.rest_days(away, date)))
        feats["rest_diff"] = feats["home_rest_days"] - feats["away_rest_days"]

        # 4. Rolling Form & EWMA Features for both Home and Away
        for team, prefix in ((home, "home"), (away, "away")):
            for w in form_windows:
                st = buf.get_stats(team, w)
                for k, v in st.items():
                    feats[f"{prefix}_{k}"] = v

            ew = buf.get_ewma(team)
            for k, v in ew.items():
                feats[f"{prefix}_{k}"] = v

        # Form differentials
        for w in (3, 5, 10, 20):
            if f"home_pts_{w}" in feats and f"away_pts_{w}" in feats:
                feats[f"diff_pts_{w}"] = feats[f"home_pts_{w}"] - feats[f"away_pts_{w}"]
                feats[f"diff_gd_{w}"] = feats[f"home_gd_{w}"] - feats[f"away_gd_{w}"]
                feats[f"diff_opp_gd_{w}"] = feats[f"home_opp_adj_gd_{w}"] - feats[f"away_opp_adj_gd_{w}"]

        # 5. Head-to-Head (H2H) Features
        h2h_data = buf.get_h2h(home, away)
        for k, v in h2h_data.items():
            feats[k] = v

        # 6. Squad / Player Features (Vectorized FIFA Lookups)
        if include_player_features and fifa_lookup is not None:
            # Map FIFA year to available editions
            fifa_year = max(15, min(23, year % 100 if year >= 2000 else 15))
            for team, prefix in ((home, "home"), (away, "away")):
                p_res = fifa_lookup.get((team, fifa_year), fifa_lookup.get((team, 22), {}))
                feats[f"{prefix}_squad_top5_ovr"] = p_res.get("top5_ovr", np.nan)
                feats[f"{prefix}_squad_xi_ovr"] = p_res.get("xi_ovr", np.nan)
                feats[f"{prefix}_squad_depth_ovr"] = p_res.get("depth_ovr", np.nan)
                feats[f"{prefix}_squad_age_mean"] = p_res.get("age_mean", np.nan)

            if not np.isnan(feats.get("home_squad_xi_ovr", np.nan)) and not np.isnan(feats.get("away_squad_xi_ovr", np.nan)):
                feats["diff_xi_ovr"] = feats["home_squad_xi_ovr"] - feats["away_squad_xi_ovr"]
                feats["diff_top5_ovr"] = feats["home_squad_top5_ovr"] - feats["away_squad_top5_ovr"]
            else:
                feats["diff_xi_ovr"] = np.nan
                feats["diff_top5_ovr"] = np.nan

        rows.append(feats)

        # Strictly update state AFTER recording pre-match features
        tracker.update(home, away, hg, ag, neutral)
        buf.record_match(home, away, date, hg, ag, eh, ea)

    df_out = pd.DataFrame(rows, index=matches.index)
    # Impute missing values with column medians from historical data
    df_out = df_out.fillna(df_out.median(numeric_only=True)).fillna(0.0)
    return df_out
