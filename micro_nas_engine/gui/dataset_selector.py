from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QFileDialog, QGroupBox, QFormLayout
)
from PySide6.QtCore import Signal

class DatasetSelector(QWidget):
    # Signal emitted when a dataset is selected, sending the path/URL
    dataset_selected = Signal(str)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Predefined Selectors
        predefined_group = QGroupBox("Select Predefined Dataset")
        predefined_layout = QHBoxLayout()
        self.predefined_combo = QComboBox()
        self.predefined_combo.addItems(["", "MNIST", "CIFAR10", "FashionMNIST"])
        self.btn_select_predefined = QPushButton("Use Predefined")
        self.btn_select_predefined.clicked.connect(self._on_predefined_selected)

        predefined_layout.addWidget(self.predefined_combo)
        predefined_layout.addWidget(self.btn_select_predefined)
        predefined_group.setLayout(predefined_layout)

        # URL Input
        url_group = QGroupBox("Enter Dataset URL (CSV)")
        url_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://example.com/data.csv")
        self.btn_select_url = QPushButton("Use URL")
        self.btn_select_url.clicked.connect(self._on_url_selected)

        url_layout.addWidget(self.url_input)
        url_layout.addWidget(self.btn_select_url)
        url_group.setLayout(url_layout)

        # Local File Input
        local_group = QGroupBox("Select Local File/Folder")
        local_layout = QHBoxLayout()
        self.local_path_label = QLabel("No path selected")
        self.btn_browse_csv = QPushButton("Browse CSV")
        self.btn_browse_folder = QPushButton("Browse Image Folder")

        self.btn_browse_csv.clicked.connect(self._on_browse_csv)
        self.btn_browse_folder.clicked.connect(self._on_browse_folder)

        local_layout.addWidget(self.local_path_label)
        local_layout.addWidget(self.btn_browse_csv)
        local_layout.addWidget(self.btn_browse_folder)
        local_group.setLayout(local_layout)

        # Add to main layout
        layout.addWidget(predefined_group)
        layout.addWidget(url_group)
        layout.addWidget(local_group)
        self.setLayout(layout)

    def _on_predefined_selected(self):
        choice = self.predefined_combo.currentText()
        if choice:
            self.dataset_selected.emit(choice)
            self.local_path_label.setText(f"Selected: {choice}")

    def _on_url_selected(self):
        url = self.url_input.text().strip()
        if url:
            self.dataset_selected.emit(url)
            self.local_path_label.setText(f"Selected URL: {url}")

    def _on_browse_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV Dataset", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.dataset_selected.emit(file_path)
            self.local_path_label.setText(f"Selected: {file_path}")

    def _on_browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Dataset Folder")
        if folder_path:
            self.dataset_selected.emit(folder_path)
            self.local_path_label.setText(f"Selected Folder: {folder_path}")
