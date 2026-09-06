import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from core.env.state import GameState, GameStatus
from core.env.action import Action
from core.env.rules import GomokuRules
from core.representation.encoder import StateEncoder
from core.representation.action_mapper import ActionMapper
from core.network.model import AlphaForgeNet
from .node import MCTSNode
from .tree import MCTSTree
from core.config import MCTS_C_PUCT, MCTS_TEMPERATURE

class MCTS:
    """
    Neural-network-guided Monte Carlo Tree Search.
    """
    def __init__(
        self, 
        network: AlphaForgeNet, 
        encoder: StateEncoder, 
        action_mapper: ActionMapper, 
        rules: GomokuRules
    ):
        self.network = network
        self.encoder = encoder
        self.action_mapper = action_mapper
        self.rules = rules

    def search(
        self, 
        root_state: GameState, 
        num_simulations: Optional[int] = None, 
        temperature: Optional[float] = None
    ) -> Tuple[Action, List[float]]:
        """
        Performs MCTS simulations and returns the best action and the search policy.
        """
        from core.config import config
        sims = num_simulations if num_simulations is not None else config.get("selfplay.sims")
        temp = temperature if temperature is not None else config.get("selfplay.temp")
        
        tree = MCTSTree(root_state)
        
        for _ in range(sims):
            # 1. Selection
            leaf, path = self._select(tree.root)
            
            # 2. Expansion & Evaluation
            value = self._expand_and_evaluate(leaf)
            
            # 3. Backup
            self._backup(path, value)
            
        # Derive search policy pi
        policy = self._get_search_policy(tree.root, temp)
        
        # Choose action (best visited)
        best_action = self._select_best_action(tree.root)
        
        return best_action, policy

    def _select(self, node: MCTSNode) -> Tuple[MCTSNode, List[MCTSNode]]:
        """Selects a leaf node using the PUCT formula."""
        path = [node]
        while not node.is_terminal and node.children:
            # If not all legal actions are expanded, we treat it as a leaf for AlphaZero
            # Actually, in AlphaZero, we expand ALL legal actions at once when we hit a leaf.
            # So if it has children, it's fully expanded.
            
            # PUCT Selection
            best_action = self._puct_select(node)
            node = node.children[best_action]
            path.append(node)
            
        return node, path

    def _puct_select(self, node: MCTSNode) -> Action:
        """Calculates Q + U and returns the action with the max value."""
        best_score = -float('inf')
        best_action = None
        
        parent_visit_sum = sum(child.visit_count for child in node.children.values())
        
        for action, child in node.children.items():
            # Q(s, a)
            q_value = child.value
            
            # U(s, a) = c_puct * P(s, a) * sqrt(sum N) / (1 + N)
            prior = node.priors[action]
            u_value = MCTS_C_PUCT * prior * (np.sqrt(parent_visit_sum) / (1 + child.visit_count))
            
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action

    def _expand_and_evaluate(self, leaf: MCTSNode) -> float:
        """Evaluates the leaf and expands its children if not terminal."""
        if leaf.is_terminal:
            # Terminal value from environment perspective
            # V = 1 (win), 0 (draw), -1 (loss)
            # The leaf.state.winner is the absolute player.
            # We need value relative to the player who just moved (parent of leaf).
            # Actually, the standard is: return value for the player to move at this node.
            return self._get_terminal_value(leaf.state)

        # 1. Encode and Predict
        encoded = self.encoder.encode(leaf.state)
        # Convert to torch tensor [1, 2, 10, 10]
        tensor_in = torch.FloatTensor(encoded).unsqueeze(0)
        
        with torch.no_grad():
            policy_logits, value_tensor = self.network(tensor_in)
            
        policy_logits = policy_logits.squeeze(0).numpy() # [100]
        value = value_tensor.item() # scalar
        
        # 2. Convert logits to priors and filter by legality
        legal_actions = leaf.state.get_legal_actions(self.rules)
        if not legal_actions:
            return 0.0 # Should not happen for non-terminal
            
        # Softmax only over legal actions
        legal_indices = [self.action_mapper.to_index(a) for a in legal_actions]
        legal_logits = policy_logits[legal_indices]
        
        # Numerically stable softmax
        exp_logits = np.exp(legal_logits - np.max(legal_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # 3. Create children
        for action, prob in zip(legal_actions, probs):
            next_state = leaf.state.apply_action(action, self.rules)
            leaf.add_child(action, next_state, prob)
            
        return value

    def _backup(self, path: List[MCTSNode], leaf_value: float):
        """Backs up the value through the path with alternating signs."""
        # leaf_value is from the perspective of the player to move at leaf.
        v = leaf_value
        for node in reversed(path):
            node.update(v)
            v = -v # Alternate perspective for the parent
            
    def _get_terminal_value(self, state: GameState) -> float:
        """Converts environment terminal result to MCTS value."""
        if state.status == GameStatus.DRAW:
            return 0.0
        
        # winner is the player who won.
        # Value is for the player to move at this state.
        # If the player to move is the winner, it's weird (terminal states don't move).
        # Usually, terminal result is relative to the player who just moved.
        # Let's be explicit:
        # If state.winner == state.current_player: return 1.0 (should not happen in normal Gomoku)
        # If state.winner == state.current_player.opponent: return -1.0
        
        winner = state.winner
        if winner == state.current_player:
            return 1.0
        elif winner == state.current_player.opponent:
            return -1.0
        return 0.0

    def _get_search_policy(self, root: MCTSNode, temperature: float) -> List[float]:
        """Converts root visit counts to a probability distribution."""
        counts = np.zeros(100)
        for action, child in root.children.items():
            idx = self.action_mapper.to_index(action)
            counts[idx] = child.visit_count
            
        if temperature == 0:
            # Argmax
            best_idx = np.argmax(counts)
            policy = np.zeros(100)
            policy[best_idx] = 1.0
            return policy.tolist()
        
        # pi(a|s) = N(s,a)^(1/tau) / sum(N(s,b)^(1/tau))
        counts = counts**(1.0 / temperature)
        sum_counts = np.sum(counts)
        if sum_counts == 0:
            return [0.0] * 100
            
        policy = counts / sum_counts
        return policy.tolist()

    def _select_best_action(self, root: MCTSNode) -> Action:
        """Returns the action with the highest visit count."""
        best_action = None
        max_visits = -1
        for action, child in root.children.items():
            if child.visit_count > max_visits:
                max_visits = child.visit_count
                best_action = action
        return best_action
