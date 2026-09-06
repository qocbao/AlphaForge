import sys
import io
import time
import os
from typing import Dict, Any, Optional, Tuple

class SystemUtils:
    
    @staticmethod
    def setup_utf8():
        if sys.stdout.encoding != 'utf-8':
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except AttributeError:
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    @staticmethod
    def clear_screen():
        """Clears the terminal screen based on the OS."""
        os.system('cls' if os.name == 'nt' else 'clear')

class ProgressReporter:
    """
    Unified runtime observability system for AlphaForge.
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self._start_time = None

    def start_timer(self):
        self._start_time = time.time()

    def get_elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def format_time(self, seconds: float) -> str:
        if seconds < 0:
            return "00:00"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def print_header(self, title: str):
        print("\n" + "═"*60)
        print(f" AlphaForge {title}")
        print("═"*60)

    def print_section(self, title: str):
        print(f"\n>>> {title}")

    def print_info(self, message: str):
        print(f" [INFO] {message}")

    def print_detail(self, message: str):
        print(f"     └─ {message}")

    def print_error(self, message: str):
        print(f" [ERROR] {message}")

    def render_board(self, board, last_move: Optional[Tuple[int, int]] = None, iteration_id: int = 0, move_count: int = 0):
        """
        Renders the board in a clean grid format with | and - symbols.
        """
        from core.env.player import Player
        # Màu ANSI
        RED = "\033[91m"
        RESET = "\033[0m"
        BOLD = "\033[1m"
        CYAN = "\033[96m"
        
        print("\n" + "═"*60)
        print(f" {BOLD}ALPHA FORGE - BOARD VIEW {RESET}")
        print(f" {CYAN}Iteration: {iteration_id} | Move: {move_count}{RESET}")
        print("═"*60)
        
        # In hàng tiêu đề cột (0 1 2 3...)
        # Mỗi ô chiếm 3 khoảng trống: " 0 |"
        header = "    " + "   ".join([f"{i}" for i in range(board.size)])
        print(header)
        
        # Đường kẻ ngang trên cùng: +---+---+---+
        top_border = "   " + "+---" * board.size + "+"
        print(top_border)
        
        for r in range(board.size):
            # Bắt đầu hàng: " 0 |"
            row_str = f"{r:2} |"
            for c in range(board.size):
                cell = board.get_cell(r, c)
                char = " " if cell == Player.EMPTY else ("X" if cell == Player.BLACK else "O")
                
                if last_move and (r, c) == last_move:
                    row_str += f" {RED}{char}{RESET} |"
                else:
                    row_str += f" {char} |"
            print(row_str)
            # Đường kẻ ngang sau mỗi hàng: +---+---+---+
            print(top_border)
            
        print("═"*60)

    def update_progress(
        self, 
        label: str, 
        current: int, 
        total: int, 
        metrics: Optional[Dict[str, Any]] = None,
        bar_width: int = 30
    ):
        if total <= 0:
            return

        percent = current / total
        filled = int(bar_width * percent)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        elapsed = self.get_elapsed()
        speed = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / speed if speed > 0 else 0
        eta_str = self.format_time(eta) if speed > 0 else "calculating..."
        
        metrics_str = ""
        if metrics:
            metrics_str = " | " + " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])

        line = f"\r{label} [{bar}] {current}/{total} ({percent:.1%}) {metrics_str} | Speed: {speed:.2f}/s | ETA: {eta_str}"
        sys.stdout.write(line[:120])
        sys.stdout.flush()

        if current == total:
            print()

    def print_summary(self, data: Dict[str, Any]):
        print("\n" + "─"*60)
        for k, v in data.items():
            print(f" {k:20}: {v}")
        print("─"*60)
