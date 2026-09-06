from typing import Tuple
from core.env.environment import GomokuEnvironment
from core.env.state import GameState, GameStatus
from core.env.player import Player
from .player import EvaluationPlayer
from .result import MatchResult, SingleMatchResult

class Match:
    """
    Orchestrates a single game between a candidate and a reference player.
    """
    def __init__(
        self, 
        env: GomokuEnvironment, 
        candidate: EvaluationPlayer, 
        reference: EvaluationPlayer
    ):
        self.env = env
        self.candidate = candidate
        self.reference = reference

    def run(self, candidate_starts: bool) -> SingleMatchResult:
        """
        Runs the game until termination.
        
        Args:
            candidate_starts: If True, candidate is BLACK (first player).
        """
        state = self.env.reset()
        
        # Track who is who relative to Gomoku's BLACK/WHITE
        # BLACK always starts in GomokuEnvironment.reset()
        if candidate_starts:
            cand_color = Player.BLACK
            ref_color = Player.WHITE
        else:
            cand_color = Player.WHITE
            ref_color = Player.BLACK
            
        while state.status == GameStatus.ONGOING:
            # Determine whose turn it is
            if state.current_player == cand_color:
                player = self.candidate
            else:
                player = self.reference
                
            # Action selection
            action = player.select_action(state)
            
            # Apply action
            state, is_terminal = self.env.step(action)
            
        # Result determination from Candidate's perspective
        result = MatchResult.DRAW
        if state.status == GameStatus.WIN:
            if state.winner == cand_color:
                result = MatchResult.CANDIDATE_WIN
            else:
                result = MatchResult.REFERENCE_WIN
                
        return SingleMatchResult(
            result=result,
            candidate_color=cand_color.name,
            move_count=state.move_count,
            winner=state.winner.name if state.winner else None
        )
