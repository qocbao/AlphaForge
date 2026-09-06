from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum, auto

class MatchResult(Enum):
    CANDIDATE_WIN = auto()
    REFERENCE_WIN = auto()
    DRAW = auto()

@dataclass(frozen=True)
class SingleMatchResult:
    """Result of a single evaluation game from the candidate's perspective."""
    result: MatchResult
    candidate_color: str # "BLACK" or "WHITE"
    move_count: int
    winner: Optional[str] # "BLACK", "WHITE", or None

@dataclass
class ArenaResult:
    """Aggregate results of an evaluation arena."""
    total_games: int
    candidate_wins: int
    reference_wins: int
    draws: int
    
    # Detailed stats
    candidate_as_first_wins: int = 0
    candidate_as_first_draws: int = 0
    candidate_as_second_wins: int = 0
    candidate_as_second_draws: int = 0
    
    def __post_init__(self):
        self._validate()

    def _validate(self):
        if self.candidate_wins + self.reference_wins + self.draws != self.total_games:
            raise ValueError("Sum of wins/losses/draws must equal total games.")

    @property
    def candidate_win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.candidate_wins / self.total_games

    @property
    def candidate_score_rate(self) -> float:
        """
        Score rate: Win = 1, Draw = 0.5, Loss = 0.
        """
        if self.total_games == 0:
            return 0.0
        return (self.candidate_wins + 0.5 * self.draws) / self.total_games

    @property
    def reference_win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.reference_wins / self.total_games

    @property
    def draw_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.draws / self.total_games
