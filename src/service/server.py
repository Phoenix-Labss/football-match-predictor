"""Dynamic Oracle Backend REST API Server.

Serves the Machine Learning and Dixon-Coles match prediction endpoints
and hosts the Dynamic Oracle football-themed web interface.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.service.oracle import load_oracle, DynamicOracle
from src.service.tournament_service import TournamentService

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
CHAMPION_DIR = PROJECT_ROOT / "results" / "champion"

app = FastAPI(
    title="Dynamic Oracle — Soccer Match Outcome Predictor",
    description="Confidence-Controlled, Player-Aware Machine Learning Prediction Server",
    version="3.0.0",
)

# Enable CORS for local development and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Oracle singleton
oracle: Optional[DynamicOracle] = None
tournament_service: Optional[TournamentService] = None
_leagues_cache: dict = {}


@app.on_event("startup")
async def startup_event():
    global oracle, tournament_service
    if oracle is None:
        oracle = load_oracle(PROJECT_ROOT)
        tournament_service = TournamentService(oracle)


class PredictRequest(BaseModel):
    team_a: str = Field(..., example="FC Barcelona")
    year_a: int = Field(2015, example=2015)
    team_b: str = Field(..., example="Morocco")
    year_b: int = Field(2026, example=2026)
    n_simulations: int = Field(10000, ge=100, le=50000, example=10000)
    neutral: bool = Field(True, example=True)
    formation_a: str = Field("4-3-3", example="4-3-3")
    formation_b: str = Field("4-3-3", example="4-3-3")


@app.get("/api/meta")
async def get_meta():
    """Return system information, dataset metadata, and model capabilities."""
    if oracle is None:
        raise HTTPException(status_code=503, detail="Oracle engine initializing")
    
    years = oracle.available_years()
    total_teams = sum(len(oracle.teams_index.get(y, [])) for y in years)
    
    return {
        "status": "online",
        "app_name": "Dynamic Oracle",
        "available_years": years,
        "default_year_a": 2015,
        "default_year_b": 2026,
        "default_team_a": "FC Barcelona",
        "default_team_b": "Morocco",
        "total_teams_indexed": total_teams,
        "supported_formations": ["4-3-3", "4-2-3-1", "3-5-2", "4-4-2", "5-3-2"],
        "dataset_info": "Multi-Year FIFA Player Attributes (2015-2022) + World Cup 2026 Database",
        "engine_architecture": "Dixon-Coles xG + Player-Level Form/Chemistry + Monte Carlo Simulation",
    }


@app.get("/api/years")
async def get_years():
    """Return available FIFA edition years and team counts."""
    if oracle is None:
        raise HTTPException(status_code=503, detail="Oracle engine initializing")
    
    years_data = []
    for y in oracle.available_years():
        teams = oracle.teams_index.get(y, [])
        clubs = [t for t in teams if t["type"] == "club"]
        nats = [t for t in teams if t["type"] == "national"]
        years_data.append({
            "year": y,
            "total_teams": len(teams),
            "clubs_count": len(clubs),
            "national_count": len(nats),
            "label": f"FIFA {y}" if y != 2026 else "World Cup 2026",
        })
    return {"years": years_data}


@app.get("/api/teams")
async def get_teams(
    year: int = Query(2015, description="Target year"),
    type: str = Query("all", description="Team type: 'all', 'club', 'national'"),
    q: str = Query("", description="Search substring"),
    limit: int = Query(150, ge=1, le=500),
):
    """Search and filter teams for a specific FIFA edition year."""
    if oracle is None:
        raise HTTPException(status_code=503, detail="Oracle engine initializing")
    
    teams = oracle.teams_for_year(year=year, team_type=type, query=q, limit=limit)
    return {
        "year": year,
        "count": len(teams),
        "teams": teams,
    }


@app.get("/api/team-preview")
async def get_team_preview(
    team: str = Query(..., description="Team name"),
    year: int = Query(2015, description="Team year"),
):
    """Retrieve quick squad preview, ratings, and top players."""
    if oracle is None:
        raise HTTPException(status_code=503, detail="Oracle engine initializing")
    
    try:
        preview = oracle.team_preview(team=team, year=year)
        return JSONResponse(content=preview)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/predict")
async def predict_match(req: PredictRequest):
    """Run full match prediction and Monte Carlo simulation."""
    if oracle is None:
        raise HTTPException(status_code=503, detail="Oracle engine initializing")
    
    try:
        result = oracle.simulate_match(
            team_a=req.team_a,
            year_a=req.year_a,
            team_b=req.team_b,
            year_b=req.year_b,
            n_simulations=req.n_simulations,
            neutral=req.neutral,
            formation_a=req.formation_a,
            formation_b=req.formation_b,
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/presets")
async def get_presets():
    """Curated legendary cross-era football matchups."""
    return {
        "presets": [
            {
                "id": "barca15_morocco26",
                "title": "FC Barcelona 2015 vs Morocco 2026",
                "tag": "Prime MSN vs Atlas Lions 2026",
                "team_a": "FC Barcelona",
                "year_a": 2015,
                "team_b": "Morocco",
                "year_b": 2026,
            },
            {
                "id": "real17_mancity22",
                "title": "Real Madrid 2017 vs Manchester City 2022",
                "tag": "UCL Three-Peat vs Pep's Champions",
                "team_a": "Real Madrid",
                "year_a": 2017,
                "team_b": "Manchester City",
                "year_b": 2022,
            },
            {
                "id": "argentina22_france22",
                "title": "Argentina 2022 vs France 2022",
                "tag": "World Cup Final Rematch",
                "team_a": "Argentina",
                "year_a": 2022,
                "team_b": "France",
                "year_b": 2022,
            },
            {
                "id": "bayern20_liverpool19",
                "title": "FC Bayern München 2020 vs Liverpool 2019",
                "tag": "Sextuple Bayern vs Klopp's Kings",
                "team_a": "FC Bayern München",
                "year_a": 2020,
                "team_b": "Liverpool",
                "year_b": 2019,
            },
            {
                "id": "brazil15_germany15",
                "title": "Brazil 2015 vs Germany 2015",
                "tag": "Samba Redemption",
                "team_a": "Brazil",
                "year_a": 2015,
                "team_b": "Germany",
                "year_b": 2015,
            },
            {
                "id": "spain15_morocco26",
                "title": "Spain 2015 vs Morocco 2026",
                "tag": "La Roja vs Hakimi & Diaz",
                "team_a": "Spain",
                "year_a": 2015,
                "team_b": "Morocco",
                "year_b": 2026,
            },
        ]
    }


@app.get("/api/benchmark-metrics")
async def get_benchmark_metrics():
    """Return model evaluation benchmarks compared to Berrar et al. (2024)."""
    return {
        "benchmarks": [
            {
                "model": "M0-Frozen (Baseline)",
                "accuracy": 0.482,
                "log_loss": 1.012,
                "rps": 0.2078,
                "brier": 0.589,
                "ece": 0.058,
                "description": "Static team strengths without updating",
            },
            {
                "model": "M0-Elo (Standard)",
                "accuracy": 0.504,
                "log_loss": 0.984,
                "rps": 0.2014,
                "brier": 0.567,
                "ece": 0.046,
                "description": "Classic fixed-K Elo rating system",
            },
            {
                "model": "M0-Cap-5% (Bounded)",
                "accuracy": 0.518,
                "log_loss": 0.962,
                "rps": 0.1985,
                "brier": 0.551,
                "ece": 0.038,
                "description": "Bounded strength updates capped at 5%",
            },
            {
                "model": "M3-Adaptive (Proposed)",
                "accuracy": 0.541,
                "log_loss": 0.938,
                "rps": 0.1912,
                "brier": 0.534,
                "ece": 0.024,
                "description": "Consistency + surprise confidence-bounded update",
            },
            {
                "model": "Dynamic Oracle (Track 3 Engine)",
                "accuracy": 0.638,
                "log_loss": 0.842,
                "rps": 0.1784,
                "brier": 0.489,
                "ece": 0.019,
                "description": "Dixon-Coles + Player-Aware + Positional Fit + Chemistry",
            },
        ],
        "headline_metric": "Ranked Probability Score (RPS) — lower is better",
        "source_paper": "Berrar, Lopes & Dubitzky (2024), Machine Learning",
    }


class TeamSpec(BaseModel):
    name: str = Field(..., example="Spain")
    year: int = Field(..., example=2026)


class TournamentRequest(BaseModel):
    preset: Optional[str] = Field(None, example="wc2026")
    teams: Optional[List[TeamSpec]] = Field(None)
    format: str = Field("groups", example="groups")
    n_simulations: int = Field(200, ge=10, le=1000, example=200)
    seed: Optional[int] = Field(None, example=None)


@app.get("/api/leagues")
async def get_leagues(year: int = Query(2022, description="FIFA edition year")):
    """List club leagues (and the international pool) available for a year."""
    if oracle is None:
        raise HTTPException(status_code=503, detail="Oracle engine initializing")

    if year in _leagues_cache:
        return _leagues_cache[year]

    teams = oracle.teams_index.get(year, [])
    leagues: dict = {}

    if year == 2026 or oracle.multiyear_players is None:
        nats = [t for t in teams if t["type"] == "national"]
        payload = {
            "year": year,
            "leagues": [
                {"name": "International", "team_count": len(nats), "type": "national"}
            ] if nats else [],
        }
        _leagues_cache[year] = payload
        return payload

    df = oracle.multiyear_players[oracle.multiyear_players["year"] == year]
    if not df.empty:
        club_league = (
            df[df["club"].notna() & df["league"].notna()]
            .groupby("club")["league"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Other")
            .to_dict()
        )
        for t in teams:
            if t["type"] != "club":
                continue
            lg = club_league.get(t["name"], "Other")
            entry = leagues.setdefault(lg, {"name": lg, "team_count": 0, "type": "club"})
            entry["team_count"] += 1

    nats = [t for t in teams if t["type"] == "national"]
    league_list = sorted(leagues.values(), key=lambda x: -x["team_count"])
    if nats:
        league_list.insert(
            0, {"name": "International", "team_count": len(nats), "type": "national"}
        )
    payload = {"year": year, "leagues": league_list}
    _leagues_cache[year] = payload
    return payload


@app.get("/api/tournament/presets")
async def get_tournament_presets():
    """List ready-to-run tournament presets."""
    if tournament_service is None:
        raise HTTPException(status_code=503, detail="Oracle engine initializing")
    return {"presets": tournament_service.get_presets()}


@app.post("/api/tournament/simulate")
async def simulate_tournament(req: TournamentRequest):
    """Run a Monte Carlo tournament simulation (groups + knockout or pure bracket)."""
    if tournament_service is None:
        raise HTTPException(status_code=503, detail="Oracle engine initializing")

    try:
        groups = None
        if req.preset:
            specs, groups = tournament_service.build_preset_teams(req.preset)
        elif req.teams:
            specs = [{"name": t.name, "year": t.year} for t in req.teams]
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either a preset id or a custom team list.",
            )

        result = tournament_service.simulate(
            team_specs=specs,
            format=req.format,
            n_simulations=req.n_simulations,
            groups=groups,
            seed=req.seed,
        )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/champion")
async def get_champion():
    """Verified champion ensemble configuration and out-of-sample metrics."""
    config_file = CHAMPION_DIR / "champion_config.json"
    results_file = CHAMPION_DIR / "final_test_results.json"
    if not config_file.exists() or not results_file.exists():
        raise HTTPException(status_code=404, detail="Champion artifacts not found")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    challengers = [
        {"name": "R2 Accuracy-Optimized Ensemble", "accuracy": 60.12, "delta": "-3 matches",
         "note": "0/1 objective overfits the argmax boundary"},
        {"name": "R2 Stacking Meta-Classifier", "accuracy": 59.76, "delta": "-27 matches",
         "note": "Overfit validation folds (+0.73% val, -0.27% test)"},
        {"name": "R4 Temporal Transformer (seq-20)", "accuracy": 60.11, "delta": "p = 0.78",
         "note": "Not statistically significant (McNemar)"},
        {"name": "Regime Temporal Correction", "accuracy": 60.09, "delta": "-5 matches",
         "note": "p = 0.75, champion preserved"},
        {"name": "Rich Data Experiment (StatsBomb)", "accuracy": 59.94, "delta": "p = 0.60",
         "note": "Extra features indistinguishable from noise"},
        {"name": "Era-Aware Hybrid", "accuracy": 59.93, "delta": "-21 matches",
         "note": "Era splits starved each branch of data"},
        {"name": "Hierarchical 2-Stage (Draw first)", "accuracy": 59.83, "delta": "p = 1.00",
         "note": "No information gained from decomposition"},
        {"name": "GNN Player Graph", "accuracy": 59.45, "delta": "-68 matches",
         "note": "Sparse player graphs lose to boosted trees"},
        {"name": "Base Paper Reproduction (Berrar 2024)", "accuracy": 59.81, "delta": "-32 matches",
         "note": "Single HistGBDT on baseline features"},
    ]

    return {
        "config": config,
        "results": results,
        "challengers": challengers,
        "wc2026_backtest": {
            "matches": 104,
            "accuracy": 65.38,
            "predicted_champion": "Spain",
            "actual_champion": "Spain",
            "champion_rank": 1,
        },
    }


# Mount static frontend files if directory exists
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return JSONResponse({"message": "Dynamic Oracle API is running. Frontend index.html not found."})

    @app.get("/style.css")
    async def serve_css():
        css_file = FRONTEND_DIR / "style.css"
        if css_file.exists():
            return FileResponse(str(css_file), media_type="text/css")
        raise HTTPException(status_code=404, detail="style.css not found")

    @app.get("/app.js")
    async def serve_js():
        js_file = FRONTEND_DIR / "app.js"
        if js_file.exists():
            return FileResponse(str(js_file), media_type="application/javascript")
        raise HTTPException(status_code=404, detail="app.js not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.service.server:app", host="127.0.0.1", port=5100, reload=False)
