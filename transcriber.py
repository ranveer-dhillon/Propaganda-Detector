"""Wraps faster-whisper to turn a stream of AudioChunk objects into
timestamped text segments.

Decoupled from audio capture and UI: it only knows how to consume
`AudioChunk`s from a queue and emit `TranscriptSegment`s via a callback,
so Phase 2 (sentiment/propaganda analysis) can subscribe to the same
callback stream without touching this module.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from faster_whisper import WhisperModel

from audio_capture import AudioChunk, TARGET_SAMPLE_RATE

# Chunks quieter than this RMS are treated as silence and skipped entirely,
# to avoid Whisper hallucinating text from near-silent audio.
SILENCE_RMS_THRESHOLD = 0.005


@dataclass
class TranscriptSegment:
    timestamp: float
    text: str

    def to_json_line(self) -> str:
        return json.dumps(asdict(self))


def _rms(samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def _longest_overlap(prev_text: str, new_text: str, max_words: int = 12) -> int:
    """Return how many leading words of new_text duplicate the trailing
    words of prev_text, so callers can trim that duplication before
    appending. Compares word by word from the end of prev / start of new.
    """
    prev_words = prev_text.split()
    new_words = new_text.split()
    max_check = min(max_words, len(prev_words), len(new_words))
    for n in range(max_check, 0, -1):
        if prev_words[-n:] == new_words[:n]:
            return n
    return 0


class Transcriber:
    def __init__(
        self,
        in_queue: "queue.Queue[AudioChunk]",
        on_segment: Callable[[TranscriptSegment], None],
        model_size: str = "base.en",
        device: str = "auto",
        compute_type: str = "default",
        transcript_file: Optional[Path] = None,
    ):
        self.in_queue = in_queue
        self.on_segment = on_segment
        self.transcript_file = transcript_file
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_tail = ""  # trailing words of the last emitted text, for dedup

        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

        if self.transcript_file is not None:
            self.transcript_file.parent.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk = self.in_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if _rms(chunk.data) < SILENCE_RMS_THRESHOLD:
                continue

            segments, _info = self.model.transcribe(
                chunk.data,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if not text:
                continue

            overlap = _longest_overlap(self._last_tail, text)
            new_words = text.split()[overlap:]
            if not new_words:
                continue
            committed_text = " ".join(new_words)

            self._last_tail = " ".join(text.split()[-12:])

            segment = TranscriptSegment(timestamp=chunk.timestamp, text=committed_text)
            self._write_to_file(segment)
            self.on_segment(segment)

    def _write_to_file(self, segment: TranscriptSegment) -> None:
        if self.transcript_file is None:
            return
        with open(self.transcript_file, "a", encoding="utf-8") as f:
            f.write(segment.to_json_line() + "\n")
