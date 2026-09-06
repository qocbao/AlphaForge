import torch
from typing import Dict, Any, Optional
from .arena import Arena
from .result import ArenaResult
from .player import EvaluationPlayer
from core.env.environment import GomokuEnvironment
from .artifact_manager import EvaluationArtifactManager

class Evaluator:
    """
    High-level interface to evaluate a candidate model against a reference model.
    """
    def __init__(
        self, 
        env_params: Optional[Dict[str, Any]] = None,
        num_games: Optional[int] = None,
        simulations: Optional[int] = None,
        promotion_threshold: Optional[float] = None,
        artifact_manager: Optional[EvaluationArtifactManager] = None
    ):
        from core.config import config
        self.env_params = env_params if env_params is not None else {
            "board_size": config.get("game.board_size"), 
            "win_length": config.get("game.win_length")
        }
        self.num_games = num_games if num_games is not None else config.get("evaluation.games")
        self.simulations = simulations if simulations is not None else config.get("evaluation.sims")
        self.promotion_threshold = promotion_threshold if promotion_threshold is not None else config.get("evaluation.promotion_threshold")
        self.artifact_manager = artifact_manager

    def _make_env(self) -> GomokuEnvironment:
        return GomokuEnvironment(**self.env_params)


    def _make_env(self) -> GomokuEnvironment:
        return GomokuEnvironment(**self.env_params)

    def evaluate(
        self, 
        candidate: EvaluationPlayer, 
        reference: EvaluationPlayer,
        reporter=None
    ) -> Dict[str, Any]:
        """
        Runs the arena and determines if the candidate should be promoted.
        """
        if reporter:
            reporter.start_timer()
            reporter.print_info(f"Evaluation: {self.num_games} games, Sims: {self.simulations}")

        arena = Arena(self._make_env, candidate, reference)
        
        # We need to inject the reporter into Arena to get live progress
        if reporter:
            # Monkey patch or modify Arena.run_evaluation to accept reporter
            # Since we can't easily change Arena's signature without updating it,
            # we'll update Arena.py next.
            result = arena.run_evaluation(self.num_games, reporter=reporter)
        else:
            result = arena.run_evaluation(self.num_games)
        
        # Promotion Decision
        score_rate = result.candidate_score_rate
        decision = "PROMOTE" if score_rate >= self.promotion_threshold else "REJECT"
        
        report = {
            "result_metrics": {
                "total_games": result.total_games,
                "candidate_wins": result.candidate_wins,
                "reference_wins": result.reference_wins,
                "draws": result.draws,
                "candidate_score_rate": result.candidate_score_rate,
            },
            "score_rate": score_rate,
            "decision": decision,
            "threshold": self.promotion_threshold,
            "games": self.num_games
        }
        
        # Persist artifacts if manager provided
        if self.artifact_manager:
            self.artifact_manager.save_evaluation(result, report)
            
        return report
