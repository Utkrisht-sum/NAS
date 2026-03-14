from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QGridLayout
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

class ResultsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Architecture Details Group
        details_group = QGroupBox("Best Architecture Discovered")
        self.grid = QGridLayout()

        # Labels setup
        self.lbl_acc = QLabel("Accuracy: N/A")
        self.lbl_params = QLabel("Parameters: N/A")
        self.lbl_flops = QLabel("FLOPs: N/A")
        self.lbl_arch_desc = QLabel("Architecture: N/A")
        self.lbl_arch_desc.setWordWrap(True)

        self.grid.addWidget(self.lbl_acc, 0, 0)
        self.grid.addWidget(self.lbl_params, 1, 0)
        self.grid.addWidget(self.lbl_flops, 2, 0)
        self.grid.addWidget(self.lbl_arch_desc, 3, 0, 1, 2)

        details_group.setLayout(self.grid)
        layout.addWidget(details_group)

        # Image Viewer Group
        img_group = QGroupBox("Visualizations")
        img_layout = QHBoxLayout()

        self.pareto_view = QLabel("Pareto Frontier will appear here")
        self.pareto_view.setAlignment(Qt.AlignCenter)
        self.pareto_view.setMinimumSize(300, 200)

        self.arch_view = QLabel("Architecture Graph will appear here")
        self.arch_view.setAlignment(Qt.AlignCenter)
        self.arch_view.setMinimumSize(300, 200)

        img_layout.addWidget(self.pareto_view)
        img_layout.addWidget(self.arch_view)
        img_group.setLayout(img_layout)

        layout.addWidget(img_group)
        self.setLayout(layout)

    def update_results(self, best_overall: dict):
        """Updates the text labels with the best architecture metrics."""
        acc = best_overall.get("accuracy", 0.0)
        params = best_overall.get("compute_cost", 0)
        flops = best_overall.get("flops", 0)
        arch = best_overall.get("architecture", {})

        self.lbl_acc.setText(f"Accuracy: {acc:.4f}")
        self.lbl_params.setText(f"Parameters: {params:,}")
        self.lbl_flops.setText(f"FLOPs (Est): {flops:,}")
        self.lbl_arch_desc.setText(f"Architecture:\n{str(arch)}")

    def update_images(self, pareto_path: str, arch_path: str):
        """Loads the generated images into the view panels."""
        if pareto_path:
            pixmap = QPixmap(pareto_path)
            scaled = pixmap.scaled(self.pareto_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.pareto_view.setPixmap(scaled)

        if arch_path:
            pixmap = QPixmap(arch_path)
            scaled = pixmap.scaled(self.arch_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.arch_view.setPixmap(scaled)
