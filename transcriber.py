# Wraps faster-whisper to turn a stream of AudioChunk objects into
# timestamped text segments.
#
# Decoupled from audio capture and UI: it only knows how to consume
# AudioChunks from a queue and emit TranscriptSegments via a callback,
# so Phase 2 (sentiment/propaganda analysis) can subscribe to the same
# callback stream without touching this module.

import json
import queue
import string
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from faster_whisper import WhisperModel

from audio_capture import AudioChunk, TARGET_SAMPLE_RATE

# Chunks quieter than this RMS are treated as silence and skipped entirely,
# to avoid Whisper hallucinating text from near-silent audio.
SILENCE_RMS_THRESHOLD = 0.005

# A gap between two consecutive words longer than this counts as a pause.
PAUSE_GAP_SECONDS = 0.5


@dataclass
class TranscriptSegment:
    timestamp: float
    text: str

    def to_json_line(self) -> str:
        return json.dumps(asdict(self))


# Running counters for one session, updated as each chunk is transcribed.
@dataclass
class SessionStats:
    total_words: int = 0
    stutter_count: int = 0
    pause_count: int = 0
    elapsed_seconds: float = 0.0

    @property
    def words_per_minute(self) -> float:
        minutes = self.elapsed_seconds / 60
        return self.total_words / minutes if minutes > 0 else 0.0


def _rms(samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


# Returns how many leading words of new_text duplicate the trailing words
# of prev_text, so callers can trim that duplication before appending.
# Compares word by word from the end of prev / start of new.
def _longest_overlap(prev_text: str, new_text: str, max_words: int = 12) -> int:
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
        on_stats: Optional[Callable[[SessionStats], None]] = None,
    ):
        self.in_queue = in_queue
        self.on_segment = on_segment
        self.on_stats = on_stats
        self.transcript_file = transcript_file
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_tail = ""  # trailing words of the last emitted text, for dedup
        self.stats = SessionStats()
        self._session_start = 0.0

        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

        if self.transcript_file is not None:
            self.transcript_file.parent.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.stats = SessionStats()
        self._session_start = time.time()
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

            try:
                self._process_chunk(chunk)
            except Exception as exc:  # keep the pipeline alive on a bad chunk
                print(f"[transcriber] error processing chunk: {exc!r}")

    def _process_chunk(self, chunk: AudioChunk) -> None:
        # Skip near-silent chunks so Whisper doesn't hallucinate text from them.
        rms = _rms(chunk.data)
        if rms < SILENCE_RMS_THRESHOLD:
            print(f"[transcriber] skipping chunk as silence (rms={rms:.5f})")
            return

        raw_segments, _info = self.model.transcribe(
            chunk.data,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
            # Greedy decoding instead of beam search: noticeably faster per
            # chunk, which matters more here than the small accuracy loss
            # since we need to keep up with a live audio stream.
            beam_size=1,
        )
        segments = list(raw_segments)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        print(f"[transcriber] rms={rms:.5f} raw_text={text!r}")
        if not text:
            return

        # Audio windows overlap, so this chunk's text likely repeats the tail
        # end of the previous chunk's text. Trim that repeated part before
        # committing, so we only emit words that are actually new.
        overlap = _longest_overlap(self._last_tail, text)
        new_words = text.split()[overlap:]
        if not new_words:
            return
        committed_text = " ".join(new_words)

        self._last_tail = " ".join(text.split()[-12:])

        self._update_stats(new_words, segments)

        segment = TranscriptSegment(timestamp=chunk.timestamp, text=committed_text)
        self._write_to_file(segment)
        self.on_segment(segment)

    # Updates the running word-count/stutter/pause stats for this chunk.
    # Stutters are consecutive duplicate words in the committed text; pauses
    # are gaps between Whisper's own (VAD-filtered) segment boundaries, so
    # this needs no extra per-word alignment pass on top of transcription.
    def _update_stats(self, new_words: List[str], segments: list) -> None:
        self.stats.total_words += len(new_words)

        for i in range(len(new_words) - 1):
            a = new_words[i].strip(string.punctuation).lower()
            b = new_words[i + 1].strip(string.punctuation).lower()
            if a and a == b:
                self.stats.stutter_count += 1

        for i in range(len(segments) - 1):
            gap = segments[i + 1].start - segments[i].end
            if gap >= PAUSE_GAP_SECONDS:
                self.stats.pause_count += 1

        self.stats.elapsed_seconds = time.time() - self._session_start
        if self.on_stats is not None:
            self.on_stats(self.stats)

    def _write_to_file(self, segment: TranscriptSegment) -> None:
        if self.transcript_file is None:
            return
        with open(self.transcript_file, "a", encoding="utf-8") as f:
            f.write(segment.to_json_line() + "\n")
