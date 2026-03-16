import sys
import os

# Ensure micro_nas_engine is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from micro_nas_engine.gui.app import run_gui

if __name__ == "__main__":
    run_gui()
