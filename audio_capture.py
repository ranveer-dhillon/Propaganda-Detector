"""WASAPI loopback audio capture.

Captures whatever the Windows default output device is playing (system audio),
not the microphone, and pushes raw float32 mono chunks onto a queue for
downstream consumption (see transcriber.py).
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

try:
    import pyaudiowpatch as pyaudio
    _BACKEND = "pyaudiowpatch"
except ImportError:  # pragma: no cover - fallback path
    pyaudio = None
    _BACKEND = None

if _BACKEND is None:
    try:
        import soundcard as sc
        _BACKEND = "soundcard"
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Neither pyaudiowpatch nor soundcard is installed. "
            "Install one of them: pip install pyaudiowpatch"
        ) from exc


TARGET_SAMPLE_RATE = 16000  # what faster-whisper expects


@dataclass
class AudioDevice:
    index: int | str
    name: str
    sample_rate: int
    channels: int


@dataclass
class AudioChunk:
    """A slice of resampled, mono, float32 PCM audio."""

    data: np.ndarray  # shape (n_samples,), dtype float32, range [-1, 1]
    timestamp: float  # time.time() when this chunk finished capturing
    sample_rate: int = TARGET_SAMPLE_RATE


def list_output_devices() -> list[AudioDevice]:
    """List loopback-capable output devices (speakers/headphones)."""
    if _BACKEND == "pyaudiowpatch":
        devices = []
        pa = pyaudio.PyAudio()
        try:
            for dev in pa.get_loopback_device_info_generator():
                devices.append(
                    AudioDevice(
                        index=dev["index"],
                        name=dev["name"],
                        sample_rate=int(dev["defaultSampleRate"]),
                        channels=dev["maxInputChannels"],
                    )
                )
        finally:
            pa.terminate()
        return devices
    else:  # soundcard
        devices = []
        for spk in sc.all_speakers():
            devices.append(
                AudioDevice(index=spk.id, name=spk.name, sample_rate=48000, channels=2)
            )
        return devices


def _resample_linear(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return data
    duration = len(data) / src_rate
    n_dst = int(round(duration * dst_rate))
    if n_dst <= 0:
        return np.zeros(0, dtype=np.float32)
    src_times = np.linspace(0, duration, num=len(data), endpoint=False)
    dst_times = np.linspace(0, duration, num=n_dst, endpoint=False)
    return np.interp(dst_times, src_times, data).astype(np.float32)


class LoopbackRecorder:
    """Continuously captures system audio and emits fixed-size overlapping
    windows onto `out_queue` as AudioChunk objects.
    """

    def __init__(
        self,
        out_queue: "queue.Queue[AudioChunk]",
        device: AudioDevice | None = None,
        window_seconds: float = 4.0,
        overlap_seconds: float = 1.0,
    ):
        self.out_queue = out_queue
        self.device = device
        self.window_seconds = window_seconds
        self.overlap_seconds = overlap_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        if _BACKEND == "pyaudiowpatch":
            self._thread = threading.Thread(target=self._run_pyaudiowpatch, daemon=True)
        else:
            self._thread = threading.Thread(target=self._run_soundcard, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    # -- pyaudiowpatch backend -------------------------------------------------

    def _run_pyaudiowpatch(self) -> None:
        pa = pyaudio.PyAudio()
        try:
            if self.device is not None:
                dev_info = pa.get_device_info_by_index(self.device.index)
            else:
                default_speakers = pa.get_default_wasapi_loopback()
                dev_info = default_speakers

            channels = int(dev_info["maxInputChannels"]) or 2
            src_rate = int(dev_info["defaultSampleRate"])
            frames_per_window = int(src_rate * self.window_seconds)
            hop = int(src_rate * (self.window_seconds - self.overlap_seconds))
            hop = max(hop, 1)

            buffer = np.zeros(0, dtype=np.float32)
            buffer_lock = threading.Lock()

            def callback(in_data, frame_count, time_info, status):
                nonlocal buffer
                samples = np.frombuffer(in_data, dtype=np.float32)
                if channels > 1:
                    samples = samples.reshape(-1, channels).mean(axis=1)
                with buffer_lock:
                    buffer = np.concatenate([buffer, samples])
                return (None, pyaudio.paContinue)

            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=src_rate,
                input=True,
                input_device_index=dev_info["index"],
                frames_per_buffer=1024,
                stream_callback=callback,
            )
            stream.start_stream()

            try:
                while not self._stop_event.is_set():
                    time.sleep(self.window_seconds - self.overlap_seconds)
                    with buffer_lock:
                        if len(buffer) < frames_per_window:
                            continue
                        window = buffer[:frames_per_window].copy()
                        buffer = buffer[hop:]
                    resampled = _resample_linear(window, src_rate, TARGET_SAMPLE_RATE)
                    self.out_queue.put(AudioChunk(data=resampled, timestamp=time.time()))
            finally:
                stream.stop_stream()
                stream.close()
        finally:
            pa.terminate()

    # -- soundcard fallback backend ---------------------------------------------

    def _run_soundcard(self) -> None:
        if self.device is not None:
            speaker = next((s for s in sc.all_speakers() if s.id == self.device.index), None)
        else:
            speaker = sc.default_speaker()
        mic = sc.get_microphone(id=speaker.id, include_loopback=True)
        src_rate = 48000
        frames_per_window = int(src_rate * self.window_seconds)
        hop = int(src_rate * (self.window_seconds - self.overlap_seconds))

        with mic.recorder(samplerate=src_rate) as rec:
            buffer = np.zeros(0, dtype=np.float32)
            while not self._stop_event.is_set():
                data = rec.record(numframes=hop)
                mono = data.mean(axis=1) if data.ndim > 1 else data
                buffer = np.concatenate([buffer, mono.astype(np.float32)])
                if len(buffer) >= frames_per_window:
                    window = buffer[:frames_per_window]
                    buffer = buffer[hop:]
                    resampled = _resample_linear(window, src_rate, TARGET_SAMPLE_RATE)
                    self.out_queue.put(AudioChunk(data=resampled, timestamp=time.time()))
