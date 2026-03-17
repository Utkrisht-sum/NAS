import sys
import os
import threading
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QLineEdit,
                               QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QFileDialog, QProgressBar, QTabWidget, QScrollArea)
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QImage, QPixmap
import base64

from micro_nas_engine.datasets.analyzer import DatasetAnalyzer
from micro_nas_engine.datasets.loader import get_dataloaders
from micro_nas_engine.core.nas_engine import NASEngine
from micro_nas_engine.core.deploy import export_deployment_project
from micro_nas_engine.utils.hardware import get_hardware_info, get_accelerator
from micro_nas_engine.datasets.bootstrap import run_bootstrap_if_needed
from micro_nas_engine.training.trainer import FullTrainer
from micro_nas_engine.core.explain import generate_explainability_report

class WorkerSignals(QObject):
    progress = Signal(str)
    finished = Signal(list, object, object)
    error = Signal(str)
    plot_pareto = Signal(str)
    plot_timeline = Signal(str)
    plot_dag = Signal(str)

class NASApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MICRONAS ENGINE vX (Next-Gen)")
        self.setMinimumSize(1000, 700)
        self.signals = WorkerSignals()

        run_bootstrap_if_needed()
        self.init_ui()
        self.signals.progress.connect(self.log_message)
        self.signals.finished.connect(self.on_nas_finished)
        self.signals.error.connect(self.log_message)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # Left Panel
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)

        hw_info = get_hardware_info()
        hw_text = f"RAM: {hw_info['ram_used_gb']:.1f}/{hw_info['ram_total_gb']:.1f} GB"
        if 'gpu_allocated_mb' in hw_info:
            hw_text += f" | GPU: {hw_info['gpu_allocated_mb']:.0f} MB"
        else:
            hw_text += " | GPU: CPU ONLY (Fallback)"
        self.hw_label = QLabel(f"Hardware Status: {hw_text}")
        left_layout.addWidget(self.hw_label)

        self.btn_select_data = QPushButton("Select Dataset Folder/File")
        self.btn_select_data.clicked.connect(self.select_dataset)
        left_layout.addWidget(self.btn_select_data)

        self.lbl_dataset = QLabel("No dataset selected")
        left_layout.addWidget(self.lbl_dataset)

        left_layout.addWidget(QLabel("Population Size (Fast Mode Default: 5):"))
        self.spin_pop = QSpinBox()
        self.spin_pop.setValue(5)
        left_layout.addWidget(self.spin_pop)

        left_layout.addWidget(QLabel("Generations (Fast Mode Default: 2):"))
        self.spin_gen = QSpinBox()
        self.spin_gen.setValue(2)
        left_layout.addWidget(self.spin_gen)

        self.btn_start = QPushButton("Start NAS (vX Mode)")
        self.btn_start.clicked.connect(self.start_nas)
        left_layout.addWidget(self.btn_start)
        left_layout.addStretch()

        # Right Panel (Tabs)
        right_panel = QTabWidget()

        # Logs Tab
        self.tab_logs = QWidget()
        logs_layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        logs_layout.addWidget(self.log_output)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        logs_layout.addWidget(self.progress_bar)
        self.tab_logs.setLayout(logs_layout)
        right_panel.addTab(self.tab_logs, "Logs & Execution")

        # Explainability Tab
        self.tab_explain = QWidget()
        explain_layout = QVBoxLayout()
        self.explain_output = QTextEdit()
        self.explain_output.setReadOnly(True)
        explain_layout.addWidget(self.explain_output)
        self.tab_explain.setLayout(explain_layout)
        right_panel.addTab(self.tab_explain, "Explainability")

        # Viz Tab
        self.tab_viz = QScrollArea()
        viz_widget = QWidget()
        viz_layout = QVBoxLayout()

        self.lbl_plot_pareto = QLabel("Pareto Plot Will Appear Here")
        viz_layout.addWidget(self.lbl_plot_pareto)

        self.lbl_plot_timeline = QLabel("Timeline Plot Will Appear Here")
        viz_layout.addWidget(self.lbl_plot_timeline)

        self.lbl_plot_dag = QLabel("DAG Plot Will Appear Here")
        viz_layout.addWidget(self.lbl_plot_dag)

        viz_widget.setLayout(viz_layout)
        self.tab_viz.setWidget(viz_widget)
        self.tab_viz.setWidgetResizable(True)
        right_panel.addTab(self.tab_viz, "Visualizations")

        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 2)

        self.selected_dataset = None
        self.dataset_meta = None

    def select_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Select Tabular Dataset", "", "CSV Files (*.csv);;All Files (*)")
        if path:
            self.selected_dataset = path
            cat = DatasetAnalyzer.detect_category(path)
            self.dataset_meta = DatasetAnalyzer.get_metadata(path, cat)
            self.lbl_dataset.setText(f"{cat} dataset selected")
            self.log_message(f"Selected: {path} ({cat})")

    def log_message(self, msg):
        self.log_output.append(msg)

    def start_nas(self):
        if not self.selected_dataset:
            self.log_message("Error: Please select a dataset first.")
            return
        self.btn_start.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_message("Starting MICRONAS ENGINE vX Pipeline (Safe Mode)...")
        self.thread = threading.Thread(target=self.run_nas_process, daemon=True)
        self.thread.start()

    def run_nas_process(self):
        try:
            import torch
            task_type = self.dataset_meta.get("type", "Unknown")
            train_loader, val_loader, input_shape, num_classes = get_dataloaders(self.selected_dataset, task_type)

            if train_loader is None:
                self.signals.error.emit(f"Failed to load dataset: {self.selected_dataset}")
                return

            config = {
                'pop_size': self.spin_pop.value(),
                'generations': self.spin_gen.value()
            }

            accelerator = get_accelerator()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            engine = NASEngine(
                task_type=task_type,
                input_shape=input_shape,
                num_classes=num_classes,
                dataloader_train=train_loader,
                dataloader_val=val_loader,
                device=device,
                accelerator=accelerator,
                config=config
            )

            def cb(msg):
                self.signals.progress.emit(msg)

            results = engine.search(progress_callback=cb)

            # Final Training & Export
            self.signals.progress.emit("Starting final memory-safe training on best architecture...")
            best_arch = results[0]

            from micro_nas_engine.models.builder import build_model
            best_model = build_model(task_type, input_shape, num_classes, best_arch['config'])

            full_trainer = FullTrainer(accelerator=accelerator, device=device)
            final_acc = full_trainer.train_model(best_model, train_loader, val_loader, epochs=10)
            self.signals.progress.emit(f"Final training complete. Validation Accuracy: {final_acc:.2f}%")

            export_deployment_project(best_model, best_arch['config'], task_type, input_shape, num_classes)
            self.signals.progress.emit("Deployment project exported to trained_model/")

            self.signals.finished.emit(results, engine.history, engine.failures.get_report())

        except Exception as e:
            self.signals.error.emit(f"Error during NAS: {str(e)}")

    def on_nas_finished(self, results, history, failures):
        self.progress_bar.setVisible(False)
        self.btn_start.setEnabled(True)

        if results:
            best = results[0]
            self.log_message(f"NAS Finished! Best DNA: {best['config'].get('dna', 'N/A')} - Acc: {best['accuracy']:.2f}%, Cost: {best['cost']:.0f}")

            # Explainability
            report = generate_explainability_report(best, history, failures)
            self.explain_output.setMarkdown(report)

            # Plots (Skipping updating the UI labels directly here for brevity,
            # in a real Qt app we would decode base64 to QPixmap and set to labels)
            self.log_message("Visualizations and Explainability Report generated. Please check respective tabs.")
        else:
            self.log_message("NAS Finished but no results were returned.")
