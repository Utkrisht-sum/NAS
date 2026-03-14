import sys
import os

# Add the current directory to sys.path so that absolute imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow
from utils.seed import set_seed
from utils.logger import setup_logger

def main():
    # Setup global app settings
    set_seed(42)
    logger = setup_logger()
    logger.info("Starting MICRONAS ENGINE...")

    # Initialize GUI
    app = QApplication(sys.path)
    window = MainWindow()
    window.show()

    # Execute App
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
