from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass(frozen=True)
class SelfPlaySample:
    """
    A single training sample generated during self-play.
    
    Contract:
    - state: Encoded board representation [2, 10, 10]
    - policy: MCTS visit-count distribution [100]
    - value: Final outcome z from current player's perspective
    """
    state: List[List[List[int]]]
    policy: List[float]
    value: float
