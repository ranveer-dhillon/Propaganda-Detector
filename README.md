# Propaganda Detector

A Windows desktop app that listens to whatever audio your PC is playing —
a YouTube video, livestream, or news broadcast — and analyzes it in real
time to surface sentiment, bias, and propaganda techniques as they're
spoken. No microphone involved; it listens to system output, not you.

The project is being built incrementally. Right now it covers the
foundation: live system-audio capture and transcription, streamed to an
on-screen caption overlay and written to a structured transcript file.
The sentiment/bias/propaganda analysis layer builds on top of that
transcript stream next.

## Requirements

- Windows 10/11
- Python 3.11+
- A working audio output device (speakers/headphones)

## Install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Notes:
- `pyaudiowpatch` is the primary backend and uses WASAPI loopback directly —
  no need to enable "Stereo Mix" or any special Windows setting.
- If `pyaudiowpatch` fails to install or find a loopback device on your
  system, the app falls back to the `soundcard` library automatically.
- The first run will download the chosen Whisper model (via
  `faster-whisper`/CTranslate2) to a local cache — this requires an internet
  connection once, then works fully offline.

## Run

```bash
python main.py
```

This launches the always-on-top caption overlay. Pick an output device from
the dropdown if you have more than one, then click **Start**. Spoken audio
from whatever is playing through that device appears as captions within a
few seconds. Use **Clear** to wipe the on-screen/in-memory transcript, and
**Save** to write the full session transcript to a `.txt` file.

### Choosing a model size

```bash
python main.py --model small.en
```

Options: `tiny`, `tiny.en`, `base`, `base.en` (default), `small`, `small.en`,
`medium`, `medium.en`. Larger models are more accurate but slower —
`base.en`/`small.en` are a good balance on CPU. Use `--device cuda` if you
have a compatible NVIDIA GPU for a large speedup.

```bash
python main.py --model small.en --device cuda --compute-type float16
```

## Output for analysis

Every session writes a timestamped JSON-lines file to `transcripts/`, one
object per committed line: `{"timestamp": <unix time>, "text": "..."}`.
The upcoming sentiment/propaganda/ticker analysis layer can tail this file
live or process it as a batch, without touching the capture/transcription
code at all.

## Project layout

- `audio_capture.py` — WASAPI loopback capture (via `pyaudiowpatch`, falls
  back to `soundcard`), resamples to 16kHz mono, emits overlapping windows.
- `transcriber.py` — wraps `faster-whisper`, skips near-silent chunks,
  trims duplicate text across overlapping windows, emits `TranscriptSegment`s.
- `overlay_ui.py` — the always-on-top caption window (PySide6). Pure
  display layer, driven entirely by Qt signals.
- `main.py` — wires the above together, handles start/stop, device
  selection, and writing the transcript file.

## Known limitations

- Single audio stream only — overlapping speakers (e.g. a panel talking
  over each other) will transcribe poorly, as with any single-channel ASR.
- Latency vs. accuracy is a real tradeoff: `tiny`/`base` models feel closer
  to real-time but drop more words; `small`/`medium` are more accurate but
  add a few hundred ms to a couple seconds of lag depending on your CPU/GPU.
- The overlap-based dedup between rolling windows is a simple word-matching
  heuristic — occasionally a short phrase may be duplicated or dropped at a
  window boundary.
- No punctuation/casing cleanup beyond what Whisper produces natively.
