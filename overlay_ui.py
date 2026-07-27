"""Always-on-top, semi-transparent live caption overlay.

Pure display layer: it only knows how to append text and expose user
actions (start/stop/clear/save/device change) via Qt signals. main.py
wires those signals to the audio capture + transcriber pipeline.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QComboBox,
    QLabel,
    QSizeGrip,
)

MAX_VISIBLE_LINES = 200


class OverlayWindow(QWidget):
    start_clicked = Signal()
    stop_clicked = Signal()
    clear_clicked = Signal()
    save_clicked = Signal()
    device_changed = Signal(int)  # index into the devices list passed to set_devices

    def __init__(self):
        super().__init__()
        self._full_transcript: list[str] = []
        self._running = False

        self.setWindowTitle("Live Captions")
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool
        )
        self.resize(720, 220)

        self.setStyleSheet(
            """
            QWidget#root {
                background-color: rgb(20, 20, 20);
            }
            QTextEdit {
                background: transparent;
                color: #f2f2f2;
                font-size: 16px;
                border: none;
            }
            QPushButton {
                background-color: rgba(255, 255, 255, 30);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 4px 10px;
            }
            QPushButton:hover { background-color: rgba(255, 255, 255, 60); }
            QComboBox {
                background-color: rgba(255, 255, 255, 30);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 2px 6px;
            }
            QLabel { color: #cccccc; }
            """
        )
        self.setObjectName("root")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)

        top_bar = QHBoxLayout()
        self.title_label = QLabel("Live Captions")
        self.device_combo = QComboBox()
        self.start_button = QPushButton("Start")
        self.clear_button = QPushButton("Clear")
        self.save_button = QPushButton("Save")
        self.close_button = QPushButton("✕")
        self.close_button.setFixedWidth(28)

        top_bar.addWidget(self.title_label)
        top_bar.addStretch(1)
        top_bar.addWidget(self.device_combo)
        top_bar.addWidget(self.start_button)
        top_bar.addWidget(self.clear_button)
        top_bar.addWidget(self.save_button)
        top_bar.addWidget(self.close_button)
        outer.addLayout(top_bar)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        outer.addWidget(self.text_area, 1)

        grip_row = QHBoxLayout()
        grip_row.addStretch(1)
        grip_row.addWidget(QSizeGrip(self))
        outer.addLayout(grip_row)

        self.start_button.clicked.connect(self._on_start_stop)
        self.clear_button.clicked.connect(self._on_clear)
        self.save_button.clicked.connect(self.save_clicked.emit)
        self.close_button.clicked.connect(self.close)
        self.device_combo.currentIndexChanged.connect(self.device_changed.emit)

        self._drag_pos = None

    # -- window dragging (frameless window has no title bar) --------------

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    # -- public API -----------------------------------------------------------

    def set_devices(self, names: list[str]) -> None:
        self.device_combo.clear()
        self.device_combo.addItems(names)

    def append_text(self, text: str) -> None:
        self._full_transcript.append(text)
        self.text_area.append(text)
        # Trim visible buffer so the widget doesn't grow unbounded.
        doc = self.text_area.document()
        if doc.blockCount() > MAX_VISIBLE_LINES:
            cursor = self.text_area.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            excess = doc.blockCount() - MAX_VISIBLE_LINES
            for _ in range(excess):
                cursor.select(cursor.SelectionType.BlockUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()
        scrollbar = self.text_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def full_transcript(self) -> str:
        return "\n".join(self._full_transcript)

    def clear_transcript(self) -> None:
        self._full_transcript.clear()
        self.text_area.clear()

    def set_running(self, running: bool) -> None:
        self._running = running
        self.start_button.setText("Stop" if running else "Start")

    # -- internal slots ---------------------------------------------------

    def _on_start_stop(self) -> None:
        if self._running:
            self.stop_clicked.emit()
        else:
            self.start_clicked.emit()

    def _on_clear(self) -> None:
        self.clear_transcript()
        self.clear_clicked.emit()
