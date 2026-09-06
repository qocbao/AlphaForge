import os
from core.utils import SystemUtils
from core.config import config

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
        
        version = config.get("project.version")
        print(f"  Version: {version} | Mode: CLI Interface")
        print("  " + "─" * 80)
        print("\n")
