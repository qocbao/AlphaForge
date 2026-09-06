import pytest
from unittest.mock import MagicMock
from core.evaluation.match import Match
from core.evaluation.result import MatchResult, SingleMatchResult
from core.env.environment import GomokuEnvironment
from core.env.state import GameState, GameStatus
from core.env.player import Player

def test_match_candidate_win():
    # Mock environment and players
    env = MagicMock()
    
    # State sequence: start -> terminal
    state_start = MagicMock()
    state_start.status = GameStatus.ONGOING
    state_start.current_player = Player.BLACK
    
    state_term = MagicMock()
    state_term.status = GameStatus.WIN
    state_term.winner = Player.BLACK
    state_term.move_count = 5
    
    env.reset.return_value = state_start
    env.step.return_value = (state_term, True)
    
    cand = MagicMock()
    cand.select_action.return_value = MagicMock()
    ref = MagicMock()
    ref.select_action.return_value = MagicMock()
    
    match = Match(env, cand, ref)
    res = match.run(candidate_starts=True)
    
    assert res.result == MatchResult.CANDIDATE_WIN
    assert res.candidate_color == "BLACK"
    assert res.move_count == 5

def test_match_reference_win():
    # Mock environment and players
    env = MagicMock()
    
    state_start = MagicMock()
    state_start.status = GameStatus.ONGOING
    state_start.current_player = Player.BLACK
    
    state_term = MagicMock()
    state_term.status = GameStatus.WIN
    state_term.winner = Player.WHITE
    state_term.move_count = 7
    
    env.reset.return_value = state_start
    env.step.return_value = (state_term, True)
    
    cand = MagicMock()
    cand.select_action.return_value = MagicMock()
    ref = MagicMock()
    ref.select_action.return_value = MagicMock()
    
    match = Match(env, cand, ref)
    res = match.run(candidate_starts=True)
    
    assert res.result == MatchResult.REFERENCE_WIN
    assert res.winner == "WHITE"
