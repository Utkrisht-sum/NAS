from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLabel
from PySide6.QtGui import QFont, QColor, QTextCursor

class TrainingLogs(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel("Training Logs & Activity")
        self.label.setFont(QFont("Arial", 10, QFont.Bold))

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Courier", 9))
        self.text_edit.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")

        layout.addWidget(self.label)
        layout.addWidget(self.text_edit)
        self.setLayout(layout)

    def append_log(self, text: str, level: str = "INFO"):
        """Appends log text with color coding based on level."""
        color_map = {
            "INFO": "#d4d4d4",   # Light grey
            "WARNING": "#d7ba7d", # Yellow/Gold
            "ERROR": "#f44747",  # Red
            "SUCCESS": "#6a9955" # Green
        }
        color = color_map.get(level.upper(), "#d4d4d4")

        # Append with HTML
        html = f'<span style="color: {color};">[{level}] {text}</span><br>'
        self.text_edit.moveCursor(QTextCursor.End)
        self.text_edit.insertHtml(html)
        self.text_edit.moveCursor(QTextCursor.End)
