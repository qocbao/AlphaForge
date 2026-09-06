from typing import List, Optional
from core.env.state import GameState
from core.env.player import Player


class StateEncoder:
    """
    Encodes a GameState into a numerical representation for neural networks.
    
    Representation Contract:
    - Shape: [channels, rows, columns] -> [2, 10, 10]
    - Channel 0: Current player's stones (1 = occupied, 0 = empty)
    - Channel 1: Opponent's stones (1 = occupied, 0 = empty)
    - Perspective: Always from the perspective of the current_player.
    - Determinism: Same state always produces same representation.
    - No Mutation: Encoding does not modify the source GameState.
    """
    def __init__(self, board_size: Optional[int] = None):
        from core.config import config
        self.board_size = board_size if board_size is not None else config.get("game.board_size")

    def encode(self, state: GameState) -> List[List[List[int]]]:
        """
        Encodes the game state into a 3D list [2, board_size, board_size].
        
        Args:
            state: The GameState to encode.
            
        Returns:
            A 3D list representing the board from the current player's perspective.
        """
        if not isinstance(state, GameState):
            raise TypeError(f"Expected GameState object, got {type(state).__name__}")

        # Current player and their opponent
        current_player = state.current_player
        opponent = current_player.opponent

        # Initialize representation [channels, rows, cols]
        # Channel 0: Current Player
        # Channel 1: Opponent
        representation = [
            [[0 for _ in range(self.board_size)] for _ in range(self.board_size)],
            [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        ]

        board = state.board
        
        # We only iterate over the board once to fill both channels
        # Use board.size to be consistent with the actual board if it differs from config,
        # but the prompt specifies 10x10.
        actual_size = board.size
        
        for r in range(actual_size):
            for c in range(actual_size):
                cell = board.get_cell(r, c)
                if cell == current_player:
                    # Ensure we don't go out of bounds if actual_size > self.board_size
                    # though they should be aligned.
                    if r < self.board_size and c < self.board_size:
                        representation[0][r][c] = 1
                elif cell == opponent:
                    if r < self.board_size and c < self.board_size:
                        representation[1][r][c] = 1
        
        return representation

    def __repr__(self) -> str:
        return f"StateEncoder(board_size={self.board_size})"
