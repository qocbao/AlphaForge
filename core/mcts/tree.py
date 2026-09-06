from typing import Optional
from .node import MCTSNode
from core.env.state import GameState

class MCTSTree:
    """
    Manages the root and overall structure of the MCTS search tree.
    """
    def __init__(self, root_state: GameState):
        self.root = MCTSNode(root_state)

    def set_root(self, node: MCTSNode):
        """Updates the root of the tree (e.g., after a move is made)."""
        self.root = node

    def reset(self, root_state: GameState):
        """Resets the tree with a new root state."""
        self.root = MCTSNode(root_state)
