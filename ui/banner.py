import os
from core.utils import setup_utf8

setup_utf8()

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_banner():
    logo = r"""
    █████╗ ██╗     ██████╗ ██╗  ██╗ █████╗ ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
   ██╔══██╗██║     ██╔══██╗██║  ██║██╔══██╗██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
   ███████║██║     ██████╔╝███████║███████║█████╗  ██║   ██║██████╔╝██║  ███╗█████╗
   ██╔══██║██║     ██╔═══╝ ██╔══██║██╔══██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
   ██║  ██║███████╗██║     ██║  ██║██║  ██║██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
   ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
    """
    return logo

def display_startup():
    clear_terminal()

    print(get_banner())

    try:
        from core.config import VERSION
    except ImportError:
        VERSION = "Unknow"

    print(f"  Version: {VERSION} | Mode: CLI Interface")
    print("  " + "─" * 80)
    print("\n")
