from typing import List, Optional, Tuple
from .player import Player
from .board import Board
from .action import Action
from .rules import GomokuRules
from .state import GameState, GameStatus

class GomokuEnvironment:
    """
    The primary interface for interacting with the Gomoku game.
    Coordinates the lifecycle of a game.
    """
    def __init__(self, board_size: int = 15, win_length: int = 5):
        self.board_size = board_size
        self.win_length = win_length
        self.rules = GomokuRules(board_size, win_length)
        self._state: Optional[GameState] = None

    def reset(self) -> GameState:
        """Resets the environment to the initial state."""
        board = Board(self.board_size)
        self._state = GameState(
            board=board,
            current_player=Player.BLACK, # Black always starts
            status=GameStatus.ONGOING,
            winner=None,
            move_count=0
        )
        return self._state

    def step(self, action: Action) -> Tuple[GameState, bool]:
        """
        Applies an action to the current state.
        Returns a tuple of (new_state, is_terminal).
        """
        if self._state is None:
            raise RuntimeError("Environment must be reset before calling step().")
        
        if self._state.status != GameStatus.ONGOING:
            raise ValueError("Game is already over. Cannot perform action.")

        # Transition to new state
        new_state = self._state.apply_action(action, self.rules)
        self._state = new_state
        
        is_terminal = (new_state.status != GameStatus.ONGOING)
        return new_state, is_terminal

    def get_current_state(self) -> GameState:
        """Returns the current GameState."""
        if self._state is None:
            raise RuntimeError("Environment must be reset before calling get_current_state().")
        return self._state

    def get_legal_actions(self) -> List[Action]:
        """Returns all legal actions for the current state."""
        if self._state is None:
            raise RuntimeError("Environment must be reset before calling get_legal_actions().")
        return self._state.get_legal_actions(self.rules)

    def get_result(self) -> Optional[Player]:
        """Returns the winner of the game, or None if ongoing or draw."""
        if self._state is None:
            return None
        return self._state.winner
