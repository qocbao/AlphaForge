from dataclasses import dataclass
from typing import List
import numpy as np
from core.config import BOARD_SIZE, ACTION_SIZE

@dataclass(frozen=True)
class ReplaySample:
    """
    A single training sample for the AlphaForge neural network.
    
    Contract:
    - state: Encoded board state [2, 10, 10]
    - policy: MCTS search policy [100]
    - value: Final outcome z from state's player perspective {-1, 0, 1}
    """
    state: List[List[List[int]]]
    policy: List[float]
    value: float

    def __post_init__(self):
        """Validate invariants of the training sample."""
        # Validate state shape [2, 10, 10]
        if len(self.state) != 2:
            raise ValueError(f"State must have 2 channels, got {len(self.state)}")
        for channel in self.state:
            if len(channel) != BOARD_SIZE:
                raise ValueError(f"State rows must be {BOARD_SIZE}, got {len(channel)}")
            for row in channel:
                if len(row) != BOARD_SIZE:
                    raise ValueError(f"State cols must be {BOARD_SIZE}, got {len(row)}")
        
        # Validate policy shape [100]
        if len(self.policy) != ACTION_SIZE:
            raise ValueError(f"Policy must have {ACTION_SIZE} actions, got {len(self.policy)}")
            
        # Validate value z in {-1, 0, 1}
        if self.value not in [-1.0, 0.0, 1.0]:
            raise ValueError(f"Value must be in {{-1, 0, 1}}, got {self.value}")
