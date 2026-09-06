import pytest
from unittest.mock import MagicMock
from core.evaluation.evaluator import Evaluator
from core.evaluation.result import ArenaResult

def test_evaluator_decision_promote():
    evaluator = Evaluator(num_games=10, promotion_threshold=0.55)
    
    # Mock EvaluationPlayer
    cand = MagicMock()
    ref = MagicMock()
    
    # Mock Arena to return a winning result
    # Arena is instantiated inside evaluate, so we patch it
    from core.evaluation.arena import Arena
    Arena.run_evaluation = MagicMock(return_value=ArenaResult(
        total_games=10,
        candidate_wins=6,
        reference_wins=2,
        draws=2
    ))
    
    report = evaluator.evaluate(cand, ref)
    
    # score_rate = (6 + 0.5*2)/10 = 0.7
    assert report["decision"] == "PROMOTE"
    assert report["score_rate"] == 0.7

def test_evaluator_decision_reject():
    evaluator = Evaluator(num_games=10, promotion_threshold=0.55)
    
    cand = MagicMock()
    ref = MagicMock()
    
    from core.evaluation.arena import Arena
    Arena.run_evaluation = MagicMock(return_value=ArenaResult(
        total_games=10,
        candidate_wins=4,
        reference_wins=5,
        draws=1
    ))
    
    report = evaluator.evaluate(cand, ref)
    
    # score_rate = (4 + 0.5*1)/10 = 0.45
    assert report["decision"] == "REJECT"
