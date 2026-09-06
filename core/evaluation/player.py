import torch
from typing import Optional
from core.network.model import AlphaForgeNet
from core.mcts.search import MCTS
from core.representation.encoder import StateEncoder
from core.representation.action_mapper import ActionMapper
from core.env.rules import GomokuRules
from core.env.state import GameState
from core.env.action import Action

class EvaluationPlayer:
    """
    An agent used during evaluation. 
    Wraps the existing MCTS and Network to provide a simple action selection interface.
    """
    def __init__(
        self, 
        model: AlphaForgeNet, 
        encoder: StateEncoder, 
        action_mapper: ActionMapper, 
        rules: GomokuRules,
        simulations: Optional[int] = None,
        temperature: Optional[float] = None
    ):
        from core.config import config
        self.model = model
        self.model.eval() # Ensure inference mode
        
        # MCTS orchestration
        self.mcts = MCTS(model, encoder, action_mapper, rules)
        self.simulations = simulations if simulations is not None else config.get("evaluation.sims", 800)
        self.temperature = temperature if temperature is not None else 0.0 # Evaluation usually uses deterministic best action

    def select_action(self, state: GameState) -> Action:
        """
        Uses MCTS to select the best action for the given state.
        """
        # MCTS.search returns (Action, policy_list)
        action, _ = self.mcts.search(
            root_state=state, 
            num_simulations=self.simulations, 
            temperature=self.temperature
        )
        return action
