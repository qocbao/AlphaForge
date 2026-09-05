# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands
- Run application: `python main.py`
- Visualize project structure: `python scripts/tree.py`

## Architecture and Structure
AlphaForge is a CLI-based AI research environment for Gomoku.

- `main.py`: Main entry point and command loop for the CLI.
- `core/`: Core logic and system utilities.
    - `config.py`: Project constants (e.g., `PROJECT_NAME`, `VERSION`).
    - `utils.py`: Low-level system utilities (e.g., UTF-8 encoding configuration).
- `ui/`: UI-related components.
    - `banner.py`: Manages terminal clearing and the ASCII startup banner.
- `scripts/`: Independent utility scripts.
    - `tree.py`: Tool for visualizing the project directory structure and saving it to `.output/project_tree.txt`.
