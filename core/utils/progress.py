import sys
import time
from typing import Optional, Callable

class ConsoleProgress:
    """
    A simple, professional console progress reporter.
    Avoids external dependencies for maximum compatibility.
    """
    def __init__(self, phase_name: str, total: int, description: str = ""):
        self.phase_name = phase_name
        self.total = total
        self.description = description
        self.start_time = None
        self.last_updated = 0

    def update(self, current: int, metrics: Optional[Dict[str, Any]] = None):
        """
        Updates the progress bar in the terminal using carriage return.
        """
        if self.start_time is None:
            self.start_time = time.time()
        
        now = time.time()
        # Update at most every 100ms to avoid terminal flickering
        if now - self.last_updated < 0.1:
            return
        self.last_updated = now

        percent = (current / self.total) * 100
        bar_length = 20
        filled_length = int(bar_length * current // self.total)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        # Calculate Speed and ETA
        elapsed = now - self.start_time
        speed = current / elapsed if elapsed > 0 else 0
        remaining = (self.total - current) / speed if speed > 0 else 0
        
        eta_str = self._format_time(remaining)
        
        # Format metrics string
        metrics_str = ""
        if metrics:
            metrics_str = " | " + " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])

        # Print the line
        sys.stdout.write(f"\r{self.phase_name} [{bar}] {current}/{self.total} ({percent:.1f}%) {metrics_str} | ETA: {eta_str}")
        sys.stdout.flush()

    def finish(self, final_metrics: Optional[Dict[str, Any]] = None):
        """Finishes the progress bar and prints a newline."""
        # Final update to ensure 100%
        self.update(self.total, final_metrics)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _format_time(self, seconds: float) -> str:
        if seconds == float('inf') or seconds < 0:
            return "calculating..."
        
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        else:
            return f"{s}s"

    def __repr__(self) -> str:
        return f"ConsoleProgress(phase={self.phase_name}, total={self.total})"
