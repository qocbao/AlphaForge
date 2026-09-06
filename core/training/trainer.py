import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple, Optional
from core.network.model import AlphaForgeNet
from core.replay.buffer import ReplayBuffer
from .loss import calculate_policy_loss, calculate_value_loss, calculate_total_loss
from .metrics import TrainingMetrics

class Trainer:
    """
    Handles the training loop for AlphaForge, consuming samples from the Replay Buffer.
    """
    def __init__(
        self, 
        model: AlphaForgeNet, 
        replay_buffer: ReplayBuffer, 
        learning_rate: float = 1e-3, 
        value_loss_weight: float = 1.0, 
        device: str = "cpu",
        metrics: Optional[TrainingMetrics] = None
    ):
        self.model = model.to(device)
        self.replay_buffer = replay_buffer
        self.value_loss_weight = value_loss_weight
        self.device = device
        self.metrics = metrics
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.current_step = 0

    def _prepare_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples from replay buffer and converts to torch tensors on the correct device.
        """
        states, policies, values = self.replay_buffer.sample(batch_size)
        
        # Convert to numpy arrays first for efficient torch tensor conversion
        # states: [B, 2, 10, 10]
        states_t = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        # policies: [B, 100]
        policies_t = torch.tensor(np.array(policies), dtype=torch.float32).to(self.device)
        # values: [B] -> [B, 1]
        values_t = torch.tensor(np.array(values), dtype=torch.float32).to(self.device).unsqueeze(1)
        
        return states_t, policies_t, values_t

    def train_step(self, batch_size: int) -> Dict[str, float]:
        """
        Performs a single gradient update step.
        
        Returns:
            A dictionary containing metrics for this step.
        """
        # 1. Sample and prepare batch
        states, policies, values = self._prepare_batch(batch_size)
        
        # 2. Set to training mode and zero gradients
        self.model.train()
        self.optimizer.zero_grad()
        
        # 3. Forward pass
        # logits: [B, 100], value_pred: [B, 1]
        logits, value_pred = self.model(states)
        
        # 4. Calculate losses
        p_loss = calculate_policy_loss(logits, policies)
        v_loss = calculate_value_loss(value_pred, values)
        total_loss, p_loss_val, v_loss_val = calculate_total_loss(
            p_loss, v_loss, self.value_loss_weight
        )
        
        # 5. Backward pass and optimizer step
        total_loss.backward()
        self.optimizer.step()
        
        self.current_step += 1
        
        # Record metrics if collector provided
        if self.metrics:
            self.metrics.add_record(
                step=self.current_step,
                total_loss=total_loss.item(),
                policy_loss=p_loss_val.item(),
                value_loss=v_loss_val.item(),
                lr=self.optimizer.param_groups[0]['lr']
            )
        
        return {
            "total_loss": total_loss.item(),
            "policy_loss": p_loss_val.item(),
            "value_loss": v_loss_val.item(),
            "step": self.current_step
        }

    def train(self, num_steps: int, batch_size: int, reporter=None) -> Dict[str, Any]:
        """
        Runs the training loop for a fixed number of steps.
        
        Returns:
            Final metrics or history.
        """
        if reporter:
            reporter.start_timer()
            reporter.print_info(f"Training config: Samples={len(self.replay_buffer)}, Batch={batch_size}, Steps={num_steps}")

        history = []
        for i in range(num_steps):
            metrics = self.train_step(batch_size)
            history.append(metrics)
            
            if reporter and i % 10 == 0:
                # Update progress bar every 10 steps to avoid terminal flooding
                reporter.update_progress(
                    label="Training",
                    current=i+1,
                    total=num_steps,
                    metrics={
                        "loss": metrics["total_loss"],
                        "p_loss": metrics["policy_loss"],
                        "v_loss": metrics["value_loss"]
                    }
                )
                
        if reporter:
            # Final update to ensure 100%
            reporter.update_progress("Training", num_steps, num_steps, metrics=history[-1])
            reporter.print_info(f"Training completed. Final Loss: {history[-1]['total_loss']:.4f}")

        return {
            "final_metrics": history[-1],
            "steps_completed": num_steps
        }
