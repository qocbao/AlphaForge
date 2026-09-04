import os
from core.utils import setup_utf8
from ui.banner import display_startup

setup_utf8()

def main():
    display_startup()
    print("Welcome to AlphaForge. The AI Research environment for Gomoku.")
    print("Type 'help' to see available commands or 'exit' to quit.")

    while True:
        try:
            cmd = input("\nAlphaForge > ").strip().lower()
            if cmd == 'exit':
                print("Shutting down AlphaForge... Goodbye!")
                break
            elif cmd == 'help':
                print("Available commands: help, exit, status")
            elif cmd == 'status':
                print("System status: All modules online. Ready for training/inference.")
            elif not cmd:
                continue
            else:
                print(f"Unknown command: '{cmd}'. Type 'help' for assistance.")
        except KeyboardInterrupt:
            print("\nShutting down AlphaForge... Goodbye!")
            break

if __name__ == "__main__":
    main()
