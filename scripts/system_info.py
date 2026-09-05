import os
import sys
import subprocess
import platform
from pathlib import Path

# Setup path to import from core
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from core.utils import SystemUtils

class SystemInfoCollector:

    def __init__(self):
        self.info = []

    def _run_powershell(self, command):
        try:
            # Using powershell -Command to get cleaner output
            full_cmd = f"powershell -NoProfile -ExecutionPolicy Bypass -Command \"{command}\""
            result = subprocess.run(full_cmd, capture_output=True, text=True, shell=True, encoding='utf-8')
            return result.stdout.strip()
        except Exception as e:
            return f"Error running command: {e}"

    def collect_os_info(self):
        self.info.append("=== Hệ Điều Hành ===")
        self.info.append(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
        self.info.append(f"Kiến trúc: {platform.machine()}")
        self.info.append("")

    def collect_cpu_info(self):
        self.info.append("=== CPU ===")
        # Get CPU info using PowerShell
        cmd = "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"
        cpu_name = self._run_powershell(cmd)

        cmd_cores = "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty NumberOfCores"
        cores = self._run_powershell(cmd_cores)

        cmd_logical = "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty NumberOfLogicalProcessors"
        logical = self._run_powershell(cmd_logical)

        self.info.append(f"Tên CPU: {cpu_name if cpu_name else 'N/A'}")
        self.info.append(f"Số nhân vật lý: {cores if cores else 'N/A'}")
        self.info.append(f"Số luồng: {logical if logical else 'N/A'}")
        self.info.append("")

    def collect_memory_info(self):
        self.info.append("=== Bộ Nhớ (RAM) ===")
        cmd = "Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty TotalPhysicalMemory"
        total_mem_raw = self._run_powershell(cmd)

        try:
            total_bytes = int(total_mem_raw)
            total_gb = total_bytes / (1024**3)
            self.info.append(f"Tổng dung lượng RAM: {total_gb:.2f} GB")
        except:
            self.info.append(f"Không thể xác định dung lượng RAM ({total_mem_raw})")
        self.info.append("")

    def collect_gpu_info(self):
        self.info.append("=== GPU (AI Hardware) ===")
        # Get all GPUs using PowerShell
        cmd_gpus = "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
        gpu_list = self._run_powershell(cmd_gpus)
        self.info.append(f"Danh sách GPU:\n{gpu_list if gpu_list else 'N/A'}")

        # Specifically check for NVIDIA
        # Check if nvidia-smi exists
        nvidia_check = subprocess.run("where nvidia-smi", capture_output=True, shell=True)
        if nvidia_check.returncode == 0:
            nvidia_info = self._run_powershell("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
            if nvidia_info:
                self.info.append("\nChi tiết NVIDIA GPU (via nvidia-smi):")
                self.info.append(nvidia_info)
        else:
            self.info.append("\nKhông tìm thấy NVIDIA GPU hoặc nvidia-smi không khả dụng.")
        self.info.append("")

    def gather_all(self):
        SystemUtils.setup_utf8()
        self.collect_os_info()
        self.collect_cpu_info()
        self.collect_memory_info()
        self.collect_gpu_info()
        return "\n".join(self.info)

    def visualize(self, save_to_file=True):
        final_output = self.gather_all()
        print(final_output)

        if save_to_file:
            output_dir = root_path / ".output"
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / "system_info.txt"

            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(final_output)
                print(f"\n* Đã lưu thông tin hệ thống vào: {output_file.relative_to(root_path)}")
            except Exception as e:
                print(f"\n* Lỗi khi lưu file: {e}")

if __name__ == "__main__":
    collector = SystemInfoCollector()
    collector.visualize()
