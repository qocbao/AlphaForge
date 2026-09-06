import pytest
from unittest.mock import MagicMock, patch
from core.self_improvement.pipeline import SelfImprovementPipeline
from core.self_improvement.result import IterationStatus

def test_pipeline_iteration_flow():
    # Mock dependencies
    encoder = MagicMock()
    action_mapper = MagicMock()
    rules = MagicMock()
    config = {
        "lr": 1e-3,
        "selfplay_games": 1,
        "train_steps": 1,
        "eval_games": 1,
        "models_dir": "test_models"
    }
    
    # Patch the expensive parts
    with patch('core.self_improvement.iteration.SelfPlayGame') as mock_game, \
         patch('core.self_improvement.iteration.Trainer') as mock_trainer, \
         patch('core.self_improvement.iteration.Evaluator') as mock_evaluator:
        
        # Setup mocks
        mock_game_instance = mock_game.return_value
        mock_game_instance.play.return_value = []
        
        mock_trainer_instance = mock_trainer.return_value
        mock_trainer_instance.train.return_value = {"final_metrics": {"loss": 0.1}}
        mock_trainer_instance.current_step = 100
        
        mock_eval_instance = mock_evaluator.return_value
        mock_eval_instance.evaluate.return_value = {"decision": "PROMOTE", "score_rate": 0.6}
        
        pipeline = SelfImprovementPipeline(encoder, action_mapper, rules, config)
        
        # We want to verify that the ModelManager inside the pipeline promotes
        with patch.object(pipeline.model_manager, 'promote_candidate') as mock_promote:
            res = pipeline.run_iteration(0)
            assert res.status == IterationStatus.COMPLETED
            assert res.promotion_decision == "PROMOTE"
            mock_promote.assert_called_once()


def test_pipeline_rejection():
    encoder = MagicMock()
    action_mapper = MagicMock()
    rules = MagicMock()
    config = {"selfplay_games": 1, "train_steps": 1, "eval_games": 1}
    
    with patch('core.self_improvement.iteration.SelfPlayGame') as mock_game, \
         patch('core.self_improvement.iteration.Trainer') as mock_trainer, \
         patch('core.self_improvement.iteration.Evaluator') as mock_evaluator:
        
        mock_game.return_value.play.return_value = []
        mock_trainer.return_value.train.return_value = {"final_metrics": {}}
        mock_trainer.return_value.current_step = 10
        mock_evaluator.return_value.evaluate.return_value = {"decision": "REJECT", "score_rate": 0.4}
        
        pipeline = SelfImprovementPipeline(encoder, action_mapper, rules, config)
        res = pipeline.run_iteration(0)
        
        assert res.status == IterationStatus.COMPLETED
        assert res.promotion_decision == "REJECT"
