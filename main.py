import os
from core.utils import SystemUtils
from ui.banner import BannerManager

class AlphaForgeCLI:
   
    def __init__(self):
        self.banner_manager = BannerManager()
        self.is_running = True

    def startup(self):
        SystemUtils.setup_utf8()

        self.banner_manager.display_startup()

        print("Welcome to AlphaForge. The AI Research environment for Gomoku.")
        print("Type 'help' to see available commands or 'exit' to quit.")

    def handle_command(self, cmd):
        cmd = cmd.strip().lower()

        if not cmd:
            return

        if cmd == 'exit':
            print("Shutting down AlphaForge... Goodbye!")
            self.is_running = False
        elif cmd == 'help':
            print("Available commands: help, exit, status")
        elif cmd == 'status':
            print("System status: All modules online. Ready for training/inference.")
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
