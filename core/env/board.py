from typing import List, Tuple, Optional
from .player import Player

class Board:
    """
    Manages the Gomoku board state.
    Responsibility: Data storage and basic cell operations.
    Does not contain win/loss logic (delegated to Rules).
    """
    def __init__(self, size: int):
        if size < 5:
            raise ValueError("Board size must be at least 5x5 for Gomoku.")
        self._size = size
        # Internal representation: a 2D list of Players
        self._cells = [[Player.EMPTY for _ in range(size)] for _ in range(size)]

    @property
    def size(self) -> int:
        return self._size

    def get_cell(self, row: int, col: int) -> Player:
        self._validate_coords(row, col)
        return self._cells[row][col]

    def set_cell(self, row: int, col: int, player: Player) -> None:
        self._validate_coords(row, col)
        if player == Player.EMPTY:
            raise ValueError("Cannot set a cell to EMPTY using set_cell. Use a clear method if needed.")
        
        # Invariant: cannot place stone on occupied cell
        if self._cells[row][col] != Player.EMPTY:
            raise ValueError(f"Cell ({row}, {col}) is already occupied by {self._cells[row][col].name}.")
        
        self._cells[row][col] = player

    def is_empty(self, row: int, col: int) -> bool:
        self._validate_coords(row, col)
        return self._cells[row][col] == Player.EMPTY

    def is_full(self) -> bool:
        for row in self._cells:
            if Player.EMPTY in row:
                return False
        return True

    def get_empty_positions(self) -> List[Tuple[int, int]]:
        """Returns a list of (row, col) tuples for all empty cells."""
        return [
            (r, c) 
            for r in range(self._size) 
            for c in range(self._size) 
            if self._cells[r][c] == Player.EMPTY
        ]

    def _validate_coords(self, row: int, col: int) -> None:
        if not (0 <= row < self._size and 0 <= col < self._size):
            raise IndexError(f"Coordinates ({row}, {col}) are out of board bounds (0-{self._size-1}).")

    def copy(self) -> 'Board':
        """Returns a deep copy of the board."""
        new_board = Board(self._size)
        new_board._cells = [row[:] for row in self._cells]
        return new_board

    def __repr__(self) -> str:
        res = [f"  {i:2}" for i in range(self._size)]
        res.append("   " + "---" * self._size)
        for r in range(self._size):
            row_str = f"{r:2}|"
            for c in range(self._size):
                char = " " if self._cells[r][c] == Player.EMPTY else ("X" if self._cells[r][c] == Player.BLACK else "O")
                row_str += f"{char} "
            res.append(row_str)
        return "\n".join(res)
