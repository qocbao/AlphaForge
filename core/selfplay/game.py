import random
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
from core.env.state import GameState, GameStatus
from core.env.action import Action
from core.env.rules import GomokuRules
from core.representation.encoder import StateEncoder
from core.representation.action_mapper import ActionMapper
from core.mcts.search import MCTS
from . import SelfPlaySample
from core.env.player import Player

class SelfPlayGame:
    """
    Coordinates a single self-play game between a model and itself.
    """
    def __init__(
        self, 
        initial_state: GameState, 
        mcts: MCTS, 
        encoder: StateEncoder, 
        rules: GomokuRules,
        action_mapper: ActionMapper,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        game_id: int = 1,
        total_games: int = 1,
        iteration_id: int = 1,
        total_iterations: int = 1
    ):
        self.current_state = initial_state
        self.mcts = mcts
        self.encoder = encoder
        self.rules = rules
        self.action_mapper = action_mapper
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.game_id = game_id
        self.total_games = total_games
        self.iteration_id = iteration_id
        self.total_iterations = total_iterations
        
        # History for target generation
        self.history: List[Tuple[List[List[List[int]]], List[float], GameState]] = []

    def play(self, progress_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None, reporter: Optional[Any] = None, iteration_id: int = 0, num_simulations: int = 800) -> List[SelfPlaySample]:
        """
        Executes the game until terminal state and returns training samples.
        """
        from core.utils import SystemUtils
        
        if reporter:
            SystemUtils.clear_screen()
            reporter.print_header(f"Self-Play Game | Iteration {self.iteration_id}/{self.total_iterations} | Game {self.game_id}/{self.total_games}")
            reporter.print_detail(f"Starting game. Initial state: {self.current_state.status}")
            reporter.render_board(self.current_state.board, iteration_id=iteration_id, move_count=self.current_state.move_count)

        while self.current_state.status == GameStatus.ONGOING:
            # 1. MCTS Search
            best_action, policy = self.mcts.search(
                self.current_state, 
                num_simulations=num_simulations,
                temperature=self.temperature
            )
            
            # 2. Record state and policy before move
            encoded_state = self.encoder.encode(self.current_state)
            self.history.append((encoded_state, policy, self.current_state))
            
            # 3. Action Selection
            action = self._sample_action(policy)
            
            # Step Environment
            self.current_state = self.current_state.apply_action(action, self.rules)
            
            if reporter:
                # CLEAR SCREEN trước khi in nước đi mới để tạo hiệu ứng cập nhật tại chỗ
                SystemUtils.clear_screen()
                reporter.print_header(f"Self-Play Game | Iteration {self.iteration_id}/{self.total_iterations} | Game {self.game_id}/{self.total_games}")
                
                player_name = "BLACK" if self.current_state.current_player.opponent == Player.BLACK else "WHITE"
                reporter.print_detail(f"Move {self.current_state.move_count}: Player {player_name} moves to ({action.row}, {action.col}) | Confidence: {max(policy):.2f}")
                
                # Render bàn cờ với nước đi mới nhất được tô màu đỏ
                reporter.render_board(self.current_state.board, last_move=(action.row, action.col), iteration_id=iteration_id, move_count=self.current_state.move_count)
            
            if progress_callback:
                progress_callback(len(self.history), {"status": "playing"})

        # 5. Determine final result z
        final_result = self._get_final_result()
        
        if reporter:
            result_str = f"Winner: {final_result}" if final_result else "Draw"
            reporter.print_detail(f"Game ended. Result: {result_str} | Total moves: {len(self.history)}")
            reporter.render_board(self.current_state.board, iteration_id=iteration_id, move_count=self.current_state.move_count)
        
        # 6. Generate samples with correct perspective
        return self._generate_samples(final_result)

    def _sample_action(self, policy: List[float]) -> Action:
        """Samples an action from the search policy π."""
        probs = np.array(policy)
        
        if np.sum(probs) == 0:
            legal = self.current_state.get_legal_actions(self.rules)
            return self.rng.choice(legal)
            
        idx = self.rng.choices(range(len(probs)), weights=probs, k=1)[0]
        return self.action_mapper.to_action(idx)

    def _get_final_result(self) -> Any:
        """Returns the absolute winner."""
        if self.current_state.status == GameStatus.DRAW:
            return None # Draw
        return self.current_state.winner

    def _generate_samples(self, final_winner) -> List[SelfPlaySample]:
        """
        Converts history into (state, pi, z) samples.
        """
        samples = []
        for encoded_state, policy, state in self.history:
            # z is from the perspective of the player to move at 'state'
            z = 0.0
            if self.current_state.status == GameStatus.WIN:
                if final_winner == state.current_player:
                    z = 1.0
                elif final_winner == state.current_player.opponent:
                    z = -1.0
            elif self.current_state.status == GameStatus.DRAW:
                z = 0.0
            
            samples.append(SelfPlaySample(
                state=encoded_state,
                policy=policy,
                value=z
            ))
        return samples
