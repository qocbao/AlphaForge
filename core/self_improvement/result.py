from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum, auto

class IterationStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()

@dataclass(frozen=True)
class IterationResult:
    """
    Structured result of one self-improvement iteration.
    """
    iteration_id: int
    reference_model_path: str
    candidate_model_path: str
    
    # Statistics from each phase
    selfplay_stats: Dict[str, Any] # e.g., {"games": 100, "samples": 12000}
    training_stats: Dict[str, Any] # e.g., {"final_loss": 0.12, "steps": 1000}
    evaluation_result: Dict[str, Any] # From Evaluator.evaluate()
    
    promotion_decision: str # "PROMOTE" or "REJECT"
    status: IterationStatus
    error_message: Optional[str] = None
