"""Track 3: FIFA-style match engine + Monte Carlo tournament simulator.

Pipeline:
    FIFA yearly player stats
        -> Player model (ability + form + age curve + availability)
        -> Squad model (starting XI, formation, positional fit, manager effect)
        -> Chemistry layer (club teammates, minutes shared, compatibility)
        -> Match engine (team ratings -> Dixon-Coles xG -> Poisson goals)
        -> Tournament simulator (World Cup format, Monte Carlo)
"""

from .player_model import PlayerModel, PlayerState
from .squad_model import SquadModel, FORMATIONS, TeamRating, ManagerStyle
from .chemistry import ChemistryModel, ChemistryConfig
from .match_engine import MatchEngine, MatchEngineConfig
from .negative_binomial_engine import NegativeBinomialEngine, NegativeBinomialConfig
from .match_day_state import MatchDayStateConfig, MatchDayStateSampler
from .tournament import WorldCupSimulator, TournamentResult

__all__ = [
    "PlayerModel",
    "PlayerState",
    "SquadModel",
    "FORMATIONS",
    "TeamRating",
    "ManagerStyle",
    "ChemistryModel",
    "ChemistryConfig",
    "MatchEngine",
    "MatchEngineConfig",
    "NegativeBinomialEngine",
    "NegativeBinomialConfig",
    "MatchDayStateConfig",
    "MatchDayStateSampler",
    "WorldCupSimulator",
    "TournamentResult",
]