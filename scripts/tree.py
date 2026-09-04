import os
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from core.utils import SystemUtils

class ProjectTreeVisualizer:
    
    def __init__(self, ignore_dirs=None):
        self.ignore_dirs = ignore_dirs or {
            '.git', '__pycache__', 'node_modules',
            '.claude', '.idea', '.vscode', '.output'
        }

    def find_project_root(self):
        current_path = Path(__file__).resolve().parent

        for parent in [current_path] + list(current_path.parents):
            if (parent / '.git').exists():
                return parent

        if current_path.name == 'scripts':
            return current_path.parent

        return current_path

    def _generate_tree_recursive(self, directory, prefix="", output_list=None):
        if output_list is None:
            output_list = []

        try:
            entries = sorted(
                list(Path(directory).iterdir()),
                key=lambda x: (x.is_file(), x.name.lower())
            )
        except PermissionError:
            return output_list

        entries = [e for e in entries if e.name not in self.ignore_dirs]

        count = len(entries)
        for i, entry in enumerate(entries):
            is_last = (i == count - 1)
            connector = "└── " if is_last else "├── "

            line = f"{prefix}{connector}{entry.name}"
            output_list.append(line)

            if entry.is_dir():
                new_prefix = prefix + ("    " if is_last else "│   ")
                self._generate_tree_recursive(entry, new_prefix, output_list)

        return output_list

    def visualize(self, save_to_file=True):
        SystemUtils.setup_utf8()
        root_path = self.find_project_root()

        tree_lines = [root_path.name + "/"]
        self._generate_tree_recursive(root_path, output_list=tree_lines)

        final_output = "\n".join(tree_lines)

        print(final_output)

        if save_to_file:
            output_dir = root_path / ".output"
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / "project_tree.txt"

            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(final_output)
                print(f"\n* Đã lưu cấu trúc thư mục vào: {output_file.relative_to(root_path)}")
            except Exception as e:
                print(f"\n* Lỗi khi lưu file: {e}")

if __name__ == "__main__":
    visualizer = ProjectTreeVisualizer()
    visualizer.visualize()
