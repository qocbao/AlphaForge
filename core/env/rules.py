from typing import List, Tuple, Optional
from .player import Player
from .board import Board
from .action import Action

class GomokuRules:
    """
    Contains the logic for Gomoku rules.
    Responsibility: Win detection, draw detection, and action validity.
    """
    def __init__(self, board_size: Optional[int] = None, win_length: Optional[int] = None):
        from core.config import config
        self.board_size = board_size if board_size is not None else config.get("game.board_size")
        self.win_length = win_length if win_length is not None else config.get("game.win_length")

    def is_valid_action(self, board: Board, action: Action) -> bool:
        """Checks if an action is valid on the current board."""
        try:
            return board.is_empty(action.row, action.col)
        except IndexError:
            return False

    def check_winner(self, board: Board, last_action: Action) -> Optional[Player]:
        """
        Checks if the last action resulted in a win.
        Returns the winning player or None if no one won.
        """
        player = board.get_cell(last_action.row, last_action.col)
        if player == Player.EMPTY:
            return None

        # Directions: (delta_row, delta_col)
        directions = [
            (0, 1),  # Horizontal
            (1, 0),  # Vertical
            (1, 1),  # Diagonal \
            (1, -1), # Diagonal /
        ]

        for dr, dc in directions:
            count = 1
            # Check forward
            count += self._count_consecutive(board, last_action.row, last_action.col, dr, dc, player)
            # Check backward
            count += self._count_consecutive(board, last_action.row, last_action.col, -dr, -dc, player)

            if count >= self.win_length:
                return player
        
        return None

    def _count_consecutive(self, board: Board, row: int, col: int, dr: int, dc: int, player: Player) -> int:
        count = 0
        r, c = row + dr, col + dc
        while 0 <= r < board.size and 0 <= c < board.size and board.get_cell(r, c) == player:
            count += 1
            r += dr
            c += dc
        return count

    def is_terminal(self, board: Board) -> bool:
        """A game is terminal if the board is full (draw) or someone has won (handled by check_winner)."""
        return board.is_full()
