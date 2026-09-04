import sys
from pathlib import Path

if sys.stdout.encoding !=  "utf-8":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        
def find_project_root():
    current_path = Path(__file__).resolve().parent
    
    for parent in [current_path] + list(current_path.parents):
        if (parent / '.git').exists():
            return parent
        
    if current_path.name == 'scripts':
        return current_path.parent

    return current_path

def print_tree(directory, prefix="") -> None:
    ignore_dirs = {'.git', '__pycache__', 'node_modules', '.claude', '.idea', '.vscode'}

    try:
        entries = sorted(list(Path(directory).iterdir()), key=lambda x: (x.is_file(), x.name.lower()))
    except PermissionError:
        return

    entries = [e for e in entries if e.name not in ignore_dirs]

    count = len(entries)
    for i, entry in enumerate(entries):
        is_last = (i == count - 1)
        connector = "└── " if is_last else "├── "

        print(f"{prefix}{connector}{entry.name}")

        if entry.is_dir():
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(entry, new_prefix)
    
if (__name__ == "__main__"):
    root_path = find_project_root()
    
    print(root_path.name + "/")
    print_tree(root_path)