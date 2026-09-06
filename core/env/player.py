from enum import Enum, auto

class Player(Enum):
    """
    Represents the players in a Gomoku game.
    """
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    @property
    def opponent(self) -> 'Player':
        """Returns the opposing player."""
        if self == Player.BLACK:
            return Player.WHITE
        if self == Player.WHITE:
            return Player.BLACK
        raise ValueError("EMPTY player does not have an opponent.")

    @classmethod
    def get_players(cls) -> list['Player']:
        """Returns a list of active players (BLACK and WHITE)."""
        return [cls.BLACK, cls.WHITE]
