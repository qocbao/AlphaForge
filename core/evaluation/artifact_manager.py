import json
import os
from typing import Dict, Any
from core.evaluation.result import ArenaResult
from core.config import config

class EvaluationArtifactManager:
    """
    Handles persistence of evaluation results and summaries.
    """
    def __init__(self, iteration: int, base_dir: str = "artifacts"):
        self.iteration = iteration
        version = config.get("project.version", "v0.0.0")
        self.iter_dir = os.path.join(base_dir, f"iterations-{version}", f"iter_{iteration:04d}")
        os.makedirs(self.iter_dir, exist_ok=True)

    def save_evaluation(self, result: ArenaResult, report: Dict[str, Any]):
        """
        Saves the arena result and the evaluator's report.
        """
        # We convert ArenaResult to a dict
        data = {
            "metrics": {
                "total_games": result.total_games,
                "candidate_wins": result.candidate_wins,
                "reference_wins": result.reference_wins,
                "draws": result.draws,
                "candidate_win_rate": result.candidate_win_rate,
                "candidate_score_rate": result.candidate_score_rate,
                "candidate_as_first_wins": result.candidate_as_first_wins,
                "candidate_as_first_draws": result.candidate_as_first_draws,
                "candidate_as_second_wins": result.candidate_as_second_wins,
                "candidate_as_second_draws": result.candidate_as_second_draws,
            },
            "report": report
        }
        
        path = os.path.join(self.iter_dir, "evaluation.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def save_summary(self, summary: Dict[str, Any]):
        """
        Saves a high-level summary of the iteration.
        """
        path = os.path.join(self.iter_dir, "summary.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4)

    def save_metadata(self, metadata: Dict[str, Any]):
        """
        Saves reproduction metadata (config, seeds, etc).
        """
        path = os.path.join(self.iter_dir, "metadata.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
