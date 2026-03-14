import os
import sys
import threading
from typing import Dict, Any

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QSpinBox, QProgressBar, QMessageBox, QGroupBox, QFormLayout
)
from PySide6.QtCore import Qt, QThread, Signal, QObject

from gui.dataset_selector import DatasetSelector
from gui.training_logs import TrainingLogs
from gui.results_panel import ResultsPanel

# Add root directory to sys.path so we can import internal modules easily
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.nas_engine import NASEngine
from visualization.pareto_plot import plot_pareto_frontier
from visualization.architecture_graph import plot_architecture_graph
from utils.logger import setup_logger, logger
from utils.config import Config

class NASWorker(QThread):
    """Background thread to run the NAS Engine without freezing the GUI."""
    progress_signal = Signal(str, str) # log message, level
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, dataset, prompt, pop_size, gens, budget):
        super().__init__()
        self.dataset = dataset
        self.prompt = prompt
        self.pop_size = pop_size
        self.gens = gens
        self.budget = budget

    def run(self):
        try:
            self.progress_signal.emit(f"Initializing NAS Engine with dataset: {self.dataset}", "INFO")

            # Create progress callback for deeper integration
            def progress_cb(gen, total_gens, ind, total_inds, status, train_loss, val_metric):
                msg = f"Gen {gen}/{total_gens} | Ind {ind}/{total_inds} | {status}"
                if train_loss > 0 or val_metric > 0:
                    msg += f" | Loss: {train_loss:.4f} | Val: {val_metric:.4f}"
                self.progress_signal.emit(msg, "INFO")

            engine = NASEngine(
                dataset_source=self.dataset,
                prompt=self.prompt,
                pop_size=self.pop_size,
                generations=self.gens,
                compute_budget=self.budget,
                progress_callback=progress_cb
            )

            results = engine.run()

            if results:
                self.progress_signal.emit("NAS completed successfully.", "SUCCESS")
                self.finished_signal.emit(results)
            else:
                self.error_signal.emit("NAS returned no valid results within budget.")

        except Exception as e:
            self.error_signal.emit(f"NAS Engine Error: {str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MICRONAS ENGINE")
        self.resize(1200, 800)
        self.current_dataset = None

        self.init_ui()
        self.apply_dark_theme()

        # Check and download initial datasets in background
        self.check_pretraining()

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout()

        # Left Panel (Controls & Progress)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setFixedWidth(400)

        # 1. Dataset Selection
        self.dataset_selector = DatasetSelector()
        self.dataset_selector.dataset_selected.connect(self.on_dataset_selected)
        left_layout.addWidget(QLabel("1. Select Dataset"))
        left_layout.addWidget(self.dataset_selector)

        # 2. Prompt Input
        prompt_group = QGroupBox("2. Task Prompt")
        prompt_layout = QVBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Train the best model for this dataset while keeping it efficient.")
        prompt_layout.addWidget(self.prompt_input)
        prompt_group.setLayout(prompt_layout)
        left_layout.addWidget(prompt_group)

        # 3. NAS Configuration
        config_group = QGroupBox("3. NAS Configuration")
        config_form = QFormLayout()

        self.spin_pop = QSpinBox()
        self.spin_pop.setRange(2, 100)
        self.spin_pop.setValue(Config.DEFAULT_POPULATION_SIZE)

        self.spin_gen = QSpinBox()
        self.spin_gen.setRange(1, 100)
        self.spin_gen.setValue(Config.DEFAULT_GENERATIONS)

        self.spin_budget = QSpinBox()
        self.spin_budget.setRange(1000, 100_000_000)
        self.spin_budget.setSingleStep(100_000)
        self.spin_budget.setValue(Config.DEFAULT_COMPUTE_BUDGET)

        config_form.addRow("Population Size:", self.spin_pop)
        config_form.addRow("Generations:", self.spin_gen)
        config_form.addRow("Max Parameters:", self.spin_budget)
        config_group.setLayout(config_form)
        left_layout.addWidget(config_group)

        # 4. Action Button & Progress
        self.btn_start = QPushButton("START NEURAL ARCHITECTURE SEARCH")
        self.btn_start.setMinimumHeight(50)
        self.btn_start.clicked.connect(self.start_nas)
        self.btn_start.setStyleSheet("background-color: #2b7042; font-weight: bold; font-size: 14px;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Indeterminate initially
        self.progress_bar.setVisible(False)

        left_layout.addWidget(self.btn_start)
        left_layout.addWidget(self.progress_bar)
        left_layout.addStretch()

        # Right Panel (Logs & Results)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        self.results_panel = ResultsPanel()
        self.logs_panel = TrainingLogs()

        right_layout.addWidget(self.results_panel, stretch=2)
        right_layout.addWidget(self.logs_panel, stretch=1)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def apply_dark_theme(self):
        dark_style = """
        QMainWindow { background-color: #2b2b2b; }
        QWidget { background-color: #2b2b2b; color: #f0f0f0; }
        QGroupBox {
            border: 1px solid #4a4a4a;
            border-radius: 5px;
            margin-top: 1ex;
            font-weight: bold;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
        QPushButton {
            background-color: #3c3f41;
            border: 1px solid #5a5a5a;
            padding: 5px;
            border-radius: 3px;
        }
        QPushButton:hover { background-color: #4b4d4f; }
        QPushButton:disabled { background-color: #202020; color: #606060; }
        QLineEdit, QSpinBox, QComboBox {
            background-color: #3c3f41;
            border: 1px solid #5a5a5a;
            padding: 4px;
            color: white;
        }
        """
        self.setStyleSheet(dark_style)

    def check_pretraining(self):
        """Runs the dataset downloader in a background thread."""
        self.logs_panel.append_log("Checking/Downloading initial pre-training datasets...", "INFO")

        def download_task():
            from datasets.dataset_loader import pretrain_download
            try:
                pretrain_download()
            except Exception as e:
                print(f"Pretraining error: {e}")

        t = threading.Thread(target=download_task)
        t.daemon = True
        t.start()

    def on_dataset_selected(self, dataset: str):
        self.current_dataset = dataset
        self.logs_panel.append_log(f"Dataset selected: {dataset}", "INFO")

    def start_nas(self):
        if not self.current_dataset:
            QMessageBox.warning(self, "Missing Configuration", "Please select a dataset first.")
            return

        prompt = self.prompt_input.text().strip()
        if not prompt:
            prompt = "Find the best architecture."

        pop_size = self.spin_pop.value()
        gens = self.spin_gen.value()
        budget = self.spin_budget.value()

        self.btn_start.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.logs_panel.append_log("\n--- Starting NAS ---", "WARNING")

        self.worker = NASWorker(self.current_dataset, prompt, pop_size, gens, budget)
        self.worker.progress_signal.connect(self.logs_panel.append_log)
        self.worker.finished_signal.connect(self.on_nas_finished)
        self.worker.error_signal.connect(self.on_nas_error)
        self.worker.start()

    def on_nas_finished(self, results: Dict[str, Any]):
        self.btn_start.setEnabled(True)
        self.progress_bar.setVisible(False)

        best = results["best_overall"]
        pareto = results["pareto_frontier"]
        all_evals = results["all_evaluated"]

        # Update results text
        self.results_panel.update_results(best)

        # Generate visualisations
        pareto_path = os.path.join(Config.BASE_DIR, "pareto_frontier.png")
        arch_path = os.path.join(Config.BASE_DIR, "architecture_diagram.png")

        plot_pareto_frontier(all_evals, pareto, save_path=pareto_path)
        plot_architecture_graph(best["architecture"], save_path=arch_path)

        # Update UI Images
        self.results_panel.update_images(pareto_path, arch_path)

    def on_nas_error(self, err_msg: str):
        self.btn_start.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.logs_panel.append_log(err_msg, "ERROR")
        QMessageBox.critical(self, "NAS Error", err_msg)
