import os
from pathlib import Path
from core.utils import setup_utf8

setup_utf8()

def find_project_root():
    current_path = Path(__file__).resolve().parent

    for parent in [current_path] + list(current_path.parents):
        if (parent / '.git').exists():
            return parent

    if current_path.name == 'scripts':
        return current_path.parent

    return current_path

def generate_tree(directory, prefix="", output_list=None):
    if output_list is None:
        output_list = []

    ignore_dirs = {'.git', '__pycache__', 'node_modules', '.claude', '.idea', '.vscode', '.output'}

    try:
        entries = sorted(list(Path(directory).iterdir()), key=lambda x: (x.is_file(), x.name.lower()))
    except PermissionError:
        return output_list

    entries = [e for e in entries if e.name not in ignore_dirs]

    count = len(entries)
    for i, entry in enumerate(entries):
        is_last = (i == count - 1)
        connector = "└── " if is_last else "├── "

        line = f"{prefix}{connector}{entry.name}"
        output_list.append(line)

        if entry.is_dir():
            new_prefix = prefix + ("    " if is_last else "│   ")
            generate_tree(entry, new_prefix, output_list)

    return output_list

if __name__ == "__main__":
    root_path = find_project_root()

    root_name = root_path.name + "/"

    tree_lines = [root_name]
    generate_tree(root_path, output_list=tree_lines)

    final_output = "\n".join(tree_lines)

    print(final_output)

    output_dir = root_path / ".output"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "project_tree.txt"

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_output)
        print(f"\n* Đã lưu cấu trúc thư mục vào: {output_file.relative_to(root_path)}")
    except Exception as e:
        print(f"\n* Lỗi khi lưu file: {e}")
