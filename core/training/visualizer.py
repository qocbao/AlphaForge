import matplotlib.pyplot as plt
import json
import os
from typing import List, Dict, Any
from .metrics import MetricRecord
from core.config import config

class TrainingVisualizer:
    """
    Generates training plots from recorded metrics.
    """
    def __init__(self, iteration: int, base_dir: str = "artifacts"):
        self.iteration = iteration
        version = config.get("project.version", "v0.0.0")
        self.iter_dir = os.path.join(base_dir, f"iterations-{version}", f"iter_{iteration:04d}")
        self.plots_dir = os.path.join(self.iter_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_metrics(self, history: List[MetricRecord]):
        """
        Generates individual plots for total loss, policy loss, value loss, and LR.
        """
        if not history:
            return

        steps = [r.step for r in history]
        metrics_to_plot = {
            "total_loss": [r.total_loss for r in history],
            "policy_loss": [r.policy_loss for r in history],
            "value_loss": [r.value_loss for r in history],
            "learning_rate": [r.learning_rate for r in history]
        }

        # Individual Plots
        for name, values in metrics_to_plot.items():
            plt.figure(figsize=(10, 6))
            plt.plot(steps, values, label=name, color='blue')
            plt.title(f"Iteration {self.iteration} - {name.replace('_', ' ').title()}")
            plt.xlabel("Step")
            plt.ylabel("Value")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            filename = f"{name}.png"
            plt.savefig(os.path.join(self.plots_dir, filename))
            plt.close()

        # Combined Plot
        plt.figure(figsize=(12, 8))
        plt.plot(steps, metrics_to_plot["total_loss"], label="Total Loss", color='black', linewidth=2)
        plt.plot(steps, metrics_to_plot["policy_loss"], label="Policy Loss", color='blue')
        plt.plot(steps, metrics_to_plot["value_loss"], label="Value Loss", color='red')
        plt.title(f"Iteration {self.iteration} - Training Overview")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, "training_overview.png"))
        plt.close()
