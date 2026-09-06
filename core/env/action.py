from dataclasses import dataclass

@dataclass(frozen=True)
class Action:
    """
    Represents a move in Gomoku.
    Immutable to ensure state safety during MCTS simulations.
    """
    row: int
    col: int

    def __repr__(self) -> str:
        return f"Action(row={self.row}, col={self.col})"
