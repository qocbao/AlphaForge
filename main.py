import os
import sys
from typing import Optional, List, Dict, Any
from core.utils import SystemUtils
from ui.banner import BannerManager

# Import các thành phần cần thiết để train
from core.representation.encoder import StateEncoder
from core.representation.action_mapper import ActionMapper
from core.env.rules import GomokuRules
from core.self_improvement.pipeline import SelfImprovementPipeline
from core.self_improvement.checkpoint import ModelManager

class AlphaForgeCLI:
    
    def __init__(self):
        self.banner_manager = BannerManager()
        self.is_running = True
        # Initialize ModelManager for utility functions
        self.model_manager = ModelManager()
        
    def startup(self):
        SystemUtils.setup_utf8()
        
        self.banner_manager.display_startup()
        
        print("Welcome to AlphaForge. The AI Research environment for Gomoku.")
        print("Type 'help' to see available commands or 'exit' to quit.")
    
    def start_training(self, device: str = "cpu", checkpoint_path: Optional[str] = None):
        """
        Khởi tạo và chạy vòng lặp tự cải tiến (Self-Improvement Loop).
        """
        from core.config import config
        
        print("\n" + "="*50)
        print(f"🚀 INITIALIZING ALPHAFORGE TRAINING PIPELINE ON {device.upper()}...")
        if checkpoint_path:
            print(f"📦 Resuming from custom checkpoint: {checkpoint_path}")
        print("="*50)
        
        # Update device in config
        config.set("training.device", device)
        
        try:
            # 1. Khởi tạo các thành phần cơ bản
            encoder = StateEncoder()
            action_mapper = ActionMapper()
            rules = GomokuRules(
                config.get("game.board_size"), 
                config.get("game.win_length")
            )
            
            # 2. Khởi tạo Pipeline
            pipeline = SelfImprovementPipeline(
                encoder=encoder,
                action_mapper=action_mapper,
                rules=rules,
                config=config,
                custom_checkpoint=checkpoint_path
            )
            
            # 3. Chạy training
            num_iterations = config.get("training.num_iterations")
            print(f"Starting {num_iterations} iterations of self-improvement...")
            
            results = pipeline.run(num_iterations=num_iterations)
            
            print("\n" + "="*50)
            print(f"✅ TRAINING COMPLETE. Processed {len(results)} iterations.")
            print("Check 'models/' and 'artifacts/' directories for results.")
            print("="*50)
            
        except Exception as e:
            print(f"\n❌ TRAINING ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

    def _handle_auto_selection(self, prompt_text: str) -> Optional[str]:
        """Helper to handle 'AUTO' selection from available checkpoints."""
        checkpoints = self.model_manager.list_all_checkpoints()
        if not checkpoints:
            print("❌ No checkpoints found in the system.")
            return None
        
        print(f"\n{prompt_text}")
        print("-" * 60)
        for i, cp in enumerate(checkpoints):
            print(f"[{i}] {cp['name']} ({cp['category']})")
        print("-" * 60)
        
        try:
            choice = input("Select index (or 'c' to cancel): ").strip()
            if choice.lower() == 'c':
                return None
            
            idx = int(choice)
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx]['path']
            else:
                print("❌ Invalid index.")
                return None
        except ValueError:
            print("❌ Please enter a valid number.")
            return None

    def handle_command(self, cmd):
        cmd = cmd.strip().lower()
        if not cmd: return

        if cmd == 'exit':
            print("Shutting down AlphaForge... Goodbye!")
            self.is_running = False
        elif cmd == 'help':
            print("\nAvailable commands:")
            print("  /train [cpu|gpu] [--checkpoint path|AUTO] - Start training")
            print("  /info [path|AUTO|best]                 - View model details")
            print("  /list                                  - List all checkpoints")
            print("  status                                 - Check system status")
            print("  exit                                    - Quit the application")
        elif cmd == 'status':
            print("System status: All modules online. Ready for training/inference.")
        elif cmd == '/list':
            checkpoints = self.model_manager.list_all_checkpoints()
            if not checkpoints:
                print("No checkpoints found.")
            else:
                print("\n--- Available Checkpoints ---")
                for cp in checkpoints:
                    print(f" {cp['path']}")
        elif cmd.startswith('/train'):
            parts = cmd.split()
            if len(parts) < 2:
                print("❌ Please specify device: /train cpu or /train gpu")
                return
            
            device = parts[1]
            if device not in ['cpu', 'gpu']:
                print("❌ Invalid device. Use 'cpu' or 'gpu'.")
                return
            
            # Handle --checkpoint
            checkpoint_path = None
            if '--checkpoint' in parts:
                try:
                    idx = parts.index('--checkpoint')
                    if idx + 1 < len(parts):
                        val = parts[idx+1].strip('"').strip("'")
                        if val.upper() == "AUTO":
                            checkpoint_path = self._handle_auto_selection("Select checkpoint to resume training:")
                        else:
                            checkpoint_path = val
                except ValueError: pass
            
            device_val = "cuda" if device == "gpu" else "cpu"
            self.start_training(device=device_val, checkpoint_path=checkpoint_path)
        elif cmd.startswith('/info'):
            parts = cmd.split()
            path = None
            if len(parts) > 1:
                val = parts[1]
                if val.upper() == "AUTO":
                    path = self._handle_auto_selection("Select model to view info:")
                elif val == "best":
                    path = self.model_manager.get_best_model_path()
                else:
                    path = val
            
            if path:
                try:
                    info = self.model_manager.get_model_info(path)
                    print("\n" + "═"*50)
                    print(f"📊 DETAILED MODEL ANALYSIS")
                    print("═"*50)
                    print(f" 📍 Path:     {info['path']}")
                    print(f" 🔢 Step:     {info['step']}")
                    print(f" 🏗 Architecture: {info['model_type']}")
                    print(f" 🏷 Version:  {info['version']}")
                    print("═"*50)
                except Exception as e:
                    print(f"❌ Error reading model info: {e}")
            else:
                print("❌ Please specify a path, 'best', or 'AUTO'.")
        else:
            print(f"Unknown command: '{cmd}'. Type 'help' for assistance.")

    def run(self):
        self.startup()
        while self.is_running:
            try:
                cmd = input("\nAlphaForge > ")
                self.handle_command(cmd)
            except KeyboardInterrupt:
                print("\nShutting down AlphaForge... Goodbye!")
                self.is_running = False
            except EOFError:
                self.is_running = False

if __name__ == "__main__":
    app = AlphaForgeCLI()
    app.run()
