from typing import List
from core.env.environment import GomokuEnvironment
from .player import EvaluationPlayer
from .match import Match
from .result import MatchResult, SingleMatchResult, ArenaResult

class Arena:
    """
    Runs multiple matches to aggregate statistical performance.
    """
    def __init__(
        self, 
        env_factory, # Callable that returns a new GomokuEnvironment
        candidate: EvaluationPlayer, 
        reference: EvaluationPlayer
    ):
        self.env_factory = env_factory
        self.candidate = candidate
        self.reference = reference

    def run_evaluation(self, num_games: int, reporter=None) -> ArenaResult:
        """
        Executes the evaluation protocol.
        """
        wins = 0
        losses = 0
        draws = 0
        
        # First-player stats for candidate
        first_wins = 0
        first_draws = 0
        second_wins = 0
        second_draws = 0
        
        for i in range(num_games):
            # Alternate player colors
            # Game 0: Candidate starts, Game 1: Reference starts...
            candidate_starts = (i % 2 == 0)
            
            env = self.env_factory()
            match = Match(env, self.candidate, self.reference)
            res = match.run(candidate_starts=candidate_starts)
            
            if res.result == MatchResult.CANDIDATE_WIN:
                wins += 1
                if candidate_starts: first_wins += 1
                else: second_wins += 1
            elif res.result == MatchResult.REFERENCE_WIN:
                losses += 1
            elif res.result == MatchResult.DRAW:
                draws += 1
                if candidate_starts: first_draws += 1
                else: second_draws += 1
            
            if reporter:
                reporter.update_progress(
                    label="Evaluation",
                    current=i+1,
                    total=num_games,
                    metrics={"Cand": wins, "Best": losses, "Draw": draws}
                )
                
        return ArenaResult(
            total_games=num_games,
            candidate_wins=wins,
            reference_wins=losses,
            draws=draws,
            candidate_as_first_wins=first_wins,
            candidate_as_first_draws=first_draws,
            candidate_as_second_wins=second_wins,
            candidate_as_second_draws=second_draws
        )
