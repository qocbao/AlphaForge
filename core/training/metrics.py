import json
import csv
import os
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from core.config import config

@dataclass
class MetricRecord:
    """
    A single training metric record.
    """
    iteration: int
    step: int
    total_loss: float
    policy_loss: float
    value_loss: float
    learning_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class TrainingMetrics:
    """
    Handles collection and persistence of training metrics.
    """
    def __init__(self, iteration: int, base_dir: str = "artifacts"):
        self.iteration = iteration
        self.history: List[MetricRecord] = []
        
        # Path setup
        version = config.get("project.version", "v0.0.0")
        self.iter_dir = os.path.join(base_dir, f"iterations-{version}", f"iter_{iteration:04d}")
                
        os.makedirs(self.iter_dir, exist_ok=True)
        self.json_path = os.path.join(self.iter_dir, "metrics.json")
        self.csv_path = os.path.join(self.iter_dir, "training.csv")

    def add_record(self, step: int, total_loss: float, policy_loss: float, value_loss: float, lr: float):
        """Adds a record to the history."""
        record = MetricRecord(
            iteration=self.iteration,
            step=step,
            total_loss=total_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            learning_rate=lr
        )
        self.history.append(record)

    def save(self):
        """Persists history to JSON and CSV."""
        data = [r.to_dict() for r in self.history]
        
        # Save JSON
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
        # Save CSV
        if not data:
            return
            
        keys = data[0].keys()
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)

    def __repr__(self) -> str:
        return f"TrainingMetrics(iteration={self.iteration}, records={len(self.history)})"
