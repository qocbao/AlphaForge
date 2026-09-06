import pytest
from core.evaluation.result import ArenaResult, MatchResult

def test_arena_result_metrics():
    # 10 games: 6 wins, 3 losses, 1 draw
    res = ArenaResult(
        total_games=10,
        candidate_wins=6,
        reference_wins=3,
        draws=1
    )
    
    assert res.candidate_win_rate == 0.6
    assert res.candidate_score_rate == (6 + 0.5*1)/10  # 0.65
    assert res.reference_win_rate == 0.3
    assert res.draw_rate == 0.1

def test_arena_result_validation():
    with pytest.raises(ValueError):
        ArenaResult(total_games=10, candidate_wins=1, reference_wins=1, draws=1)

def test_zero_games():
    res = ArenaResult(total_games=0, candidate_wins=0, reference_wins=0, draws=0)
    assert res.candidate_win_rate == 0.0
    assert res.candidate_score_rate == 0.0
