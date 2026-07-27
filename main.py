"""Phase 1: Real-Time System Audio Transcriber.

Wires together audio_capture (WASAPI loopback) -> transcriber
(faster-whisper) -> overlay_ui (live caption window), plus writing the
running transcript to disk for later phases to consume.
"""

from __future__ import annotations

import argparse
import queue
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QFileDialog

from audio_capture import AudioChunk, LoopbackRecorder, get_default_loopback_device, list_output_devices
from overlay_ui import OverlayWindow
from transcriber import Transcriber, TranscriptSegment

TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"


class _SegmentBridge(QObject):
    """Relays TranscriptSegments from the transcriber's worker thread to
    the Qt UI thread (Qt auto-queues signal delivery across threads).
    """

    new_text = Signal(str)


class App:
    def __init__(self, model_size: str, device_arg: str, compute_type: str):
        self.model_size = model_size
        self.device_arg = device_arg
        self.compute_type = compute_type

        self.devices = list_output_devices()
        default_device = get_default_loopback_device()
        default_index = 0
        if default_device is not None:
            for i, d in enumerate(self.devices):
                if d.index == default_device.index:
                    default_index = i
                    break
        self.selected_device = self.devices[default_index] if self.devices else None
        print(f"[main] default output device: {self.selected_device.name if self.selected_device else None!r}")

        self.audio_queue: "queue.Queue[AudioChunk]" = queue.Queue(maxsize=50)
        self.recorder: LoopbackRecorder | None = None
        self.transcriber: Transcriber | None = None

        self.bridge = _SegmentBridge()
        self.window = OverlayWindow()
        self.window.set_devices([d.name for d in self.devices] or ["Default output"])
        self.window.device_combo.setCurrentIndex(default_index)

        self.window.start_clicked.connect(self.start)
        self.window.stop_clicked.connect(self.stop)
        self.window.clear_clicked.connect(self._on_clear)
        self.window.save_clicked.connect(self._on_save)
        self.window.device_changed.connect(self._on_device_changed)
        self.bridge.new_text.connect(self.window.append_text)

    def _on_device_changed(self, idx: int) -> None:
        if 0 <= idx < len(self.devices):
            self.selected_device = self.devices[idx]

    def _on_clear(self) -> None:
        pass  # overlay already clears its own display + in-memory transcript

    def _on_save(self) -> None:
        default_name = f"transcript_{datetime.now():%Y%m%d_%H%M%S}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self.window, "Save Transcript", default_name, "Text files (*.txt)"
        )
        if path:
            Path(path).write_text(self.window.full_transcript(), encoding="utf-8")

    def start(self) -> None:
        self.audio_queue = queue.Queue(maxsize=50)

        TRANSCRIPTS_DIR.mkdir(exist_ok=True)
        session_file = TRANSCRIPTS_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.jsonl"

        self.transcriber = Transcriber(
            in_queue=self.audio_queue,
            on_segment=self._on_segment,
            model_size=self.model_size,
            device=self.device_arg,
            compute_type=self.compute_type,
            transcript_file=session_file,
        )
        self.transcriber.start()

        self.recorder = LoopbackRecorder(
            out_queue=self.audio_queue, device=self.selected_device
        )
        self.recorder.start()

        self.window.set_running(True)

    def stop(self) -> None:
        if self.recorder:
            self.recorder.stop()
            self.recorder = None
        if self.transcriber:
            self.transcriber.stop()
            self.transcriber = None
        self.window.set_running(False)

    def _on_segment(self, segment: TranscriptSegment) -> None:
        # Called from the transcriber's background thread.
        self.bridge.new_text.emit(segment.text)

    def shutdown(self) -> None:
        self.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time system audio transcriber")
    parser.add_argument(
        "--model",
        default="base.en",
        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en"],
        help="faster-whisper model size (default: base.en)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Compute device for faster-whisper: cpu, cuda, or auto (default: cpu; "
        "cuda/auto require the CUDA toolkit's cuBLAS/cuDNN DLLs to be installed)",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        help="faster-whisper compute_type, e.g. int8, float16, default",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qt_app = QApplication(sys.argv)

    app = App(model_size=args.model, device_arg=args.device, compute_type=args.compute_type)
    qt_app.aboutToQuit.connect(app.shutdown)
    app.window.show()

    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
