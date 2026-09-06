import pytest
from unittest.mock import MagicMock
from core.evaluation.arena import Arena
from core.evaluation.result import MatchResult

def test_arena_color_alternation():
    # Mock dependencies
    env_factory = MagicMock(return_value=MagicMock())
    cand = MagicMock()
    ref = MagicMock()
    
    # We mock the Match class internally or just check if the 
    # logic for alternating candidate_starts is correct.
    # To truly test the alternation, we can mock Match.run.
    
    from core.evaluation.match import Match
    Match.run = MagicMock(return_value=MagicMock(result=MatchResult.DRAW))
    
    arena = Arena(env_factory, cand, ref)
    arena.run_evaluation(num_games=10)
    
    # Verify Match.run was called 10 times
    assert Match.run.call_count == 10
    
    # Verify the 'candidate_starts' argument alternated
    calls = Match.run.call_args_list
    starts = [call.kwargs['candidate_starts'] for call in calls]
    assert starts == [True, False] * 5

def test_arena_aggregation():
    env_factory = MagicMock(return_value=MagicMock())
    cand = MagicMock()
    ref = MagicMock()
    
    from core.evaluation.match import Match
    # Mock results: 3 wins, 2 draws, 5 losses for candidate
    results = (
        [MatchResult.CANDIDATE_WIN] * 3 + 
        [MatchResult.DRAW] * 2 + 
        [MatchResult.REFERENCE_WIN] * 5
    )
    
    def mock_run(self, env, c, r, candidate_starts=True):
        # This is tricky because Match is instantiated inside run_evaluation
        # We use a side_effect on the Match.run method
        pass

    # Better way: patch Match.run
    Match.run = MagicMock(side_effect=[
        MagicMock(result=res, candidate_color="B", move_count=10, winner="B") 
        for res in results
    ])
    
    arena = Arena(env_factory, cand, ref)
    res = arena.run_evaluation(num_games=10)
    
    assert res.candidate_wins == 3
    assert res.draws == 2
    assert res.reference_wins == 5
    assert res.candidate_score_rate == (3 + 0.5*2)/10 # 0.4
