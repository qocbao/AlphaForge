import torch
from typing import List, Dict, Any, Optional, Tuple
from core.network.model import AlphaForgeNet
from core.representation.encoder import StateEncoder
from core.representation.action_mapper import ActionMapper
from core.env.rules import GomokuRules
from core.training.checkpoint import load_checkpoint
from .checkpoint import ModelManager
from .iteration import Iteration
from .result import IterationResult, IterationStatus

class SelfImprovementPipeline:
    """
    High-level orchestrator for the AlphaForge self-improvement loop.
    """
    def __init__(
        self, 
        encoder: StateEncoder, 
        action_mapper: ActionMapper, 
        rules: GomokuRules,
        config: Dict[str, Any],
        custom_checkpoint: Optional[str] = None
    ):
        self.encoder = encoder
        self.action_mapper = action_mapper
        self.rules = rules
        self.config = config
        self.model_manager = ModelManager(base_dir=config.get("paths.models_dir"))
        self.custom_checkpoint = custom_checkpoint
        
    def _get_current_best(self) -> Tuple[AlphaForgeNet, torch.optim.Optimizer]:
        """Loads the latest accepted best model."""
        model = AlphaForgeNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.get("training.lr"))
        
        # load_checkpoint will fail if file missing, we handle that in run()
        try:
            step = self.model_manager.load_best_model(model, optimizer)
        except FileNotFoundError:
            # If no best model exists, we return the random one
            # But we should probably initialize it first
            pass
            
        return model, optimizer

    def run_iteration(self, iteration_id: int, num_iterations: int) -> IterationResult:
        """
        Executes one full iteration of the loop.
        """
        # Load model and optimizer using config values
        model = AlphaForgeNet()
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.config.get("training.lr")
        )
        
        # Load the latest accepted best model or a custom checkpoint
        try:
            if self.custom_checkpoint:
                self.model_manager.load_custom_model(model, optimizer, self.custom_checkpoint)
            else:
                self.model_manager.load_best_model(model, optimizer)
        except (FileNotFoundError, Exception) as e:
            # If no best model exists or custom path is wrong, we return the random one
            pass
            
        iteration = Iteration(
            iteration_id=iteration_id,
            total_iterations=num_iterations,
            model_manager=self.model_manager,
            encoder=self.encoder,
            action_mapper=self.action_mapper,
            rules=self.rules,
            config=self.config
        )
        
        return iteration.run(model, optimizer)

    def run(self, num_iterations: int, start_iteration: int = 0) -> List[IterationResult]:
        """
        Runs the self-improvement loop for N iterations.
        """
        from core.utils import ProgressReporter
        reporter = ProgressReporter()
        
        reporter.print_header("Self-Improvement Pipeline")
        
        results = []
        for i in range(start_iteration, start_iteration + num_iterations):
            reporter.print_section(f"Iteration {i+1} / {num_iterations}")
            
            # Time the iteration
            reporter.start_timer()
            res = self.run_iteration(i, num_iterations)
            elapsed = reporter.get_elapsed()
            
            results.append(res)
            
            if res.status == IterationStatus.COMPLETED:
                # Summary for the iteration
                summary = {
                    "Status": "COMPLETED",
                    "Self-Play": f"{res.selfplay_stats.get('games')} games",
                    "Samples": res.selfplay_stats.get("samples"),
                    "Final Loss": res.training_stats.get("total_loss"),
                    "Eval Score": f"{res.evaluation_result.get('score_rate'):.2%}",
                    "Decision": res.promotion_decision,
                    "Time": reporter.format_time(elapsed)
                }
                reporter.print_summary(summary)
            elif res.status == IterationStatus.FAILED:
                reporter.print_error(f"Iteration {i+1} failed: {res.error_message}")
                break
                
        reporter.print_header("Run Complete")
        
        # Final Run Summary
        promoted = sum(1 for r in results if r.promotion_decision == "PROMOTE")
        final_summary = {
            "Total Iterations": len(results),
            "Promoted Models": promoted,
            "Rejected Models": len(results) - promoted,
            "Best Model Path": self.model_manager.get_best_model_path()
        }
        reporter.print_summary(final_summary)
        
        return results
