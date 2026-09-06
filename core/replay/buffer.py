from typing import List, Tuple, Optional
import random
import numpy as np
from collections import deque
from .sample import ReplaySample

class ReplayBuffer:
    """
    A finite replay buffer for storing and sampling AlphaZero training data.
    
    Implements FIFO replacement policy and uniform random sampling.
    """
    def __init__(self, capacity: int, seed: Optional[int] = None):
        if capacity <= 0:
            raise ValueError("Buffer capacity must be a positive integer.")
        
        self._capacity = capacity
        self._buffer = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def add(self, sample: ReplaySample) -> None:
        """Adds a single sample to the buffer. FIFO replacement is handled by deque."""
        if not isinstance(sample, ReplaySample):
            raise TypeError(f"Expected ReplaySample, got {type(sample).__name__}")
        
        # Store a copy to prevent external mutation if the sample contained mutable arrays
        # ReplaySample is frozen, but we ensure internal lists are snapshots
        # Since the encoder returns lists and state/policy are lists, we'll treat them as immutable
        # in the context of the frozen dataclass, but for absolute safety:
        self._buffer.append(sample)

    def extend(self, samples: List[ReplaySample]) -> None:
        """Adds multiple samples to the buffer."""
        for s in samples:
            self.add(s)

    def sample(self, batch_size: int) -> Tuple[List[List[List[List[int]]]], List[List[float]], List[float]]:
        """
        Samples a random batch of distinct samples.
        
        Returns:
            A tuple of (states, policies, values).
            - states: [B, 2, 10, 10]
            - policies: [B, 100]
            - values: [B] (or [B, 1] depending on preference, here returning [B])
        """
        if len(self._buffer) == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
            
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Batch size {batch_size} exceeds current buffer size {len(self._buffer)}."
            )
            
        # Sample without replacement
        batch = self._rng.sample(self._buffer, batch_size)
        
        # Unzip samples into batches
        states = [s.state for s in batch]
        policies = [s.policy for s in batch]
        values = [s.value for s in batch]
        
        return states, policies, values

    def clear(self) -> None:
        """Removes all samples from the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        """Returns the number of currently stored samples."""
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        return self._capacity if hasattr(self, '_capacity') else self._buffer.maxlen

    def __repr__(self) -> str:
        return f"ReplayBuffer(len={len(self)}, capacity={self._buffer.maxlen})"
