import torch
import numpy as np
import copy
import datetime
from typing import Dict, Any, Tuple
from core.env.environment import GomokuEnvironment
from core.env.state import GameState
from core.env.rules import GomokuRules
from core.representation.encoder import StateEncoder
from core.representation.action_mapper import ActionMapper
from core.mcts.search import MCTS
from core.selfplay.game import SelfPlayGame
from core.replay.buffer import ReplayBuffer
from core.replay.sample import ReplaySample
from core.training.trainer import Trainer
from core.training.metrics import TrainingMetrics
from core.training.visualizer import TrainingVisualizer
from core.evaluation.evaluator import Evaluator
from core.evaluation.player import EvaluationPlayer
from core.evaluation.artifact_manager import EvaluationArtifactManager
from .result import IterationResult, IterationStatus
from .checkpoint import ModelManager

class Iteration:
    """
    Executes one complete self-improvement iteration and manages its artifacts.
    """
    def __init__(self, 
        iteration_id: int,
        total_iterations: int,
        model_manager: ModelManager,
        encoder: StateEncoder,
        action_mapper: ActionMapper,
        rules: GomokuRules,
        config: Dict[str, Any]
    ):
        self.iteration_id = iteration_id
        self.total_iterations = total_iterations
        self.model_manager = model_manager
        self.encoder = encoder
        self.action_mapper = action_mapper
        self.rules = rules
        self.config = config
        
        # Artifact Managers
        self.metrics_collector = TrainingMetrics(iteration=iteration_id)
        self.eval_artifact_manager = EvaluationArtifactManager(iteration=iteration_id)
        self.visualizer = TrainingVisualizer(iteration=iteration_id)

    def run(self, current_best_model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> IterationResult:
        """
        Runs: Self-Play -> Replay -> Training -> Evaluation -> Promotion.
        """
        from core.utils import ProgressReporter, SystemUtils
        reporter = ProgressReporter()
        
        try:
            # 1. Self-Play
            reporter.print_info("Phase 1/5: Self-Play")
            reporter.print_info(f"Generating {self.config.get('selfplay.games')} games of self-play...")
            reporter.start_timer()
            
            mcts = MCTS(current_best_model, self.encoder, self.action_mapper, self.rules)
            env = GomokuEnvironment(
                self.config.get("game.board_size"), 
                self.config.get("game.win_length")
            )
            
            num_games = self.config.get("selfplay.games")
            all_samples = []
            for i in range(num_games):
                game = SelfPlayGame(
                    initial_state=env.reset(),
                    mcts=mcts,
                    encoder=self.encoder,
                    rules=self.rules,
                    action_mapper=self.action_mapper,
                    temperature=self.config.get("selfplay.temp"),
                    game_id=i + 1,
                    total_games=num_games,
                    iteration_id=self.iteration_id + 1,
                    total_iterations=self.total_iterations
                )
                # Pass the sims from config to game.play
                all_samples.extend(game.play(
                    reporter=reporter, 
                    iteration_id=self.iteration_id+1,
                    num_simulations=self.config.get("selfplay.sims")
                ))
                
                # Update progress after each game
                reporter.update_progress(
                    label="Self-Play",
                    current=i+1,
                    total=num_games,
                    metrics={"samples": len(all_samples)}
                )
            
            # 2. Replay
            reporter.print_info("Phase 2/5: Replay Buffer")
            buffer = ReplayBuffer(capacity=len(all_samples) + 100)
            for s in all_samples:
                buffer.add(ReplaySample(state=s.state, policy=s.policy, value=s.value))
            
            # 3. Training
            reporter.print_info("Phase 3/5: Training")
            candidate_model = copy.deepcopy(current_best_model)
            candidate_optimizer = torch.optim.AdamW(
                candidate_model.parameters(), 
                lr=self.config.get("training.lr", 1e-3)
            )
            
            trainer = Trainer(
                model=candidate_model,
                replay_buffer=buffer,
                learning_rate=self.config.get("training.lr"),
                device=self.config.get("training.device"),
                metrics=self.metrics_collector
            )
            
            train_metrics = trainer.train(
                num_steps=self.config.get("training.train_steps"),
                batch_size=self.config.get("training.batch_size"),
                reporter=reporter
            )
            
            self.metrics_collector.save()
            self.visualizer.plot_metrics(self.metrics_collector.history)
            
            # Save candidate/checkpoint liền sau Training (Flow chuẩn)
            candidate_path = self.model_manager.save_candidate(
                candidate_model, candidate_optimizer, self.iteration_id, trainer.current_step
            )

            # 4. Evaluation
            reporter.print_info("Phase 4/5: Evaluation")
            evaluator = Evaluator(
                num_games=self.config.get("evaluation.games"),
                simulations=self.config.get("evaluation.sims"),
                promotion_threshold=self.config.get("evaluation.promotion_threshold"),
                artifact_manager=self.eval_artifact_manager
            )
            
            cand_player = EvaluationPlayer(candidate_model, self.encoder, self.action_mapper, self.rules)
            ref_player = EvaluationPlayer(current_best_model, self.encoder, self.action_mapper, self.rules)
            
            # Quan trọng: Biến eval_report phải được gán ngay sau khi evaluate xong
            eval_report = evaluator.evaluate(cand_player, ref_player, reporter=reporter)
            
            # 5. Summary
            decision = eval_report["decision"]
            
            summary = {
                "iteration": self.iteration_id,
                "parent_model": self.model_manager.get_best_model_path(),
                "candidate_model": candidate_path,
                "training_final_loss": train_metrics["final_metrics"].get("total_loss"),
                "evaluation_score": eval_report["score_rate"],
                "decision": decision
            }
            self.eval_artifact_manager.save_summary(summary)
            
            metadata = {
                "timestamp": datetime.datetime.now().isoformat(),
                "config": self.config.data,
                "iteration": self.iteration_id
            }
            self.eval_artifact_manager.save_metadata(metadata)
            
            # Bootstrap: Promote the first model regardless of performance to establish a baseline
            is_first_model = not self.model_manager.best_model_exists()
            
            if decision == "PROMOTE" or is_first_model:
                if is_first_model:
                    reporter.print_info("First model detected. Bootstrapping best_model.pt...")
                self.model_manager.promote_candidate(candidate_path)
                
            return IterationResult(
                iteration_id=self.iteration_id,
                reference_model_path=self.model_manager.get_best_model_path(),
                candidate_model_path=candidate_path,
                selfplay_stats={"games": num_games, "samples": len(all_samples)},
                training_stats=train_metrics["final_metrics"],
                evaluation_result=eval_report,
                promotion_decision=decision,
                status=IterationStatus.COMPLETED
            )
            
        except Exception as e:
            reporter.print_error(f"Iteration failed: {str(e)}")
            return IterationResult(
                iteration_id=self.iteration_id,
                reference_model_path=self.model_manager.get_best_model_path(),
                candidate_model_path="",
                selfplay_stats={},
                training_stats={},
                evaluation_result={},
                promotion_decision="FAILED",
                status=IterationStatus.FAILED,
                error_message=str(e)
            )