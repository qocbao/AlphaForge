from typing import Optional
from core.env.action import Action


class ActionMapper:
    """
    Handles bidirectional mapping between Gomoku Actions and Policy Indices.
    
    Contract:
    - Board: 10x10
    - Policy Space: 0...99
    - Mapping: index = row * 10 + col
    """
    def __init__(self, board_size: Optional[int] = None):
        from core.config import config
        self.board_size = board_size if board_size is not None else config.get("game.board_size")
        self.action_space = self.board_size * self.board_size

    def to_index(self, action: Action) -> int:
        """
        Converts an Action(row, col) to a policy index.
        
        Args:
            action: The Action to map.
            
        Returns:
            The corresponding policy index.
            
        Raises:
            ValueError: If action coordinates are out of bounds.
        """
        if not isinstance(action, Action):
            raise TypeError(f"Expected Action object, got {type(action).__name__}")

        if not (0 <= action.row < self.board_size and 0 <= action.col < self.board_size):
            raise ValueError(
                f"Action coordinates ({action.row}, {action.col}) "
                f"are out of bounds for board size {self.board_size}."
            )
            
        return action.row * self.board_size + action.col

    def to_action(self, index: int) -> Action:
        """
        Converts a policy index back to an Action(row, col).
        
        Args:
            index: The policy index to map.
            
        Returns:
            The corresponding Action object.
            
        Raises:
            ValueError: If the index is outside the valid policy space.
        """
        if not (0 <= index < self.action_space):
            raise ValueError(
                f"Policy index {index} is out of bounds for action space {self.action_space}."
            )
            
        row = index // self.board_size
        col = index % self.board_size
        return Action(row, col)

    def __repr__(self) -> str:
        return f"ActionMapper(board_size={self.board_size}, action_space={self.action_space})"
