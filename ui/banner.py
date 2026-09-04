import os
from core.utils import SystemUtils
from core.config import VERSION

class BannerManager:

    def clear_terminal(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def get_banner(self):
        logo = r"""
    █████╗ ██╗     ██████╗ ██╗  ██╗ █████╗ ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
   ██╔══██╗██║     ██╔══██╗██║  ██║██╔══██╗██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
   ███████║██║     ██████╔╝███████║███████║█████╗  ██║   ██║██████╔╝██║  ███╗█████╗
   ██╔══██║██║     ██╔═══╝ ██╔══██║██╔══██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
   ██║  ██║███████╗██║     ██║  ██║██║  ██║██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
   ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
        """
        return logo

    def display_startup(self):
        SystemUtils.setup_utf8()
        self.clear_terminal()

        print(self.get_banner())

        print(f"  Version: {VERSION} | Mode: CLI Interface")
        print("  " + "─" * 80)
        print("\n")
