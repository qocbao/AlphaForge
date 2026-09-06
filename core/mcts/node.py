from typing import Dict, Optional, List
from core.env.state import GameState, GameStatus
from core.env.action import Action

class MCTSNode:
    """
    Represents a single state in the MCTS search tree.
    """
    def __init__(
        self, 
        state: GameState, 
        parent: Optional['MCTSNode'] = None, 
        action_from_parent: Optional[Action] = None
    ):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        
        # Visit statistics
        self.visit_count = 0
        self.value_sum = 0.0
        
        # Children: Action -> MCTSNode
        self.children: Dict[Action, 'MCTSNode'] = {}
        
        # Prior probabilities: Action -> float
        self.priors: Dict[Action, float] = {}
        
        # Terminal status
        self.is_terminal = (state.status != GameStatus.ONGOING)
        self.terminal_value: Optional[float] = None

    @property
    def value(self) -> float:
        """Returns the mean value Q(s,a) = W / N."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, action_priors: Dict[Action, float]):
        """
        Expands the node by creating children for all provided action priors.
        """
        for action, prior in action_priors.items():
            # The state for the child is obtained via state.apply_action
            # We assume this is called with rules provided by the search.
            # Since the node doesn't own rules, the search.py will likely
            # handle the state transition and just call add_child.
            pass

    def add_child(self, action: Action, state: GameState, prior: float):
        """Adds a child node to the tree."""
        child_node = MCTSNode(state, parent=self, action_from_parent=action)
        self.children[action] = child_node
        self.priors[action] = prior

    def update(self, value: float):
        """Updates the node's visit count and value sum."""
        self.visit_count += 1
        self.value_sum += value

    def __repr__(self) -> str:
        return f"MCTSNode(visits={self.visit_count}, value={self.value:.3f}, children={len(self.children)})"
