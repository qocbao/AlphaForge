from dataclasses import dataclass
from typing import List, Optional
from .player import Player
from .board import Board
from .action import Action
from .rules import GomokuRules

class GameStatus:
    ONGOING = "ONGOING"
    WIN = "WIN"
    DRAW = "DRAW"

@dataclass(frozen=True)
class GameState:
    """
    Represents the entire state of a Gomoku game at a given moment.
    Immutable to support MCTS simulations (state isolation).
    """
    board: Board
    current_player: Player
    status: str
    winner: Optional[Player]
    move_count: int

    def apply_action(self, action: Action, rules: GomokuRules) -> 'GameState':
        """
        Transitions to a new state by applying an action.
        Returns a NEW GameState instance.
        """
        # Validate action
        if not rules.is_valid_action(self.board, action):
            raise ValueError(f"Invalid action {action} for current state.")

        # Create a new board and apply the move
        new_board = self.board.copy()
        new_board.set_cell(action.row, action.col, self.current_player)

        # Check for winner
        winner = rules.check_winner(new_board, action)
        
        # Determine new status
        if winner:
            new_status = GameStatus.WIN
        elif new_board.is_full():
            new_status = GameStatus.DRAW
        else:
            new_status = GameStatus.ONGOING

        return GameState(
            board=new_board,
            current_player=self.current_player.opponent,
            status=new_status,
            winner=winner,
            move_count=self.move_count + 1
        )

    def get_legal_actions(self, rules: GomokuRules) -> List[Action]:
        """Returns all valid actions for the current player."""
        if self.status != GameStatus.ONGOING:
            return []
        
        return [Action(r, c) for r, c in self.board.get_empty_positions()]
