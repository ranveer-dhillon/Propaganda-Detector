# Propaganda Detector

A Windows desktop app that listens to whatever audio your PC is playing.
A YouTube video, livestream, or news broadcast — and analyzes it in real
time to surface sentiment, bias, and propaganda techniques as they're
spoken. No microphone involved; it listens to system output, not you.

The project is being built incrementally. Right now it covers the
foundation: live system-audio capture and transcription, streamed to an
on-screen caption overlay and written to a structured transcript file.
The sentiment/bias/propaganda analysis layer builds on top of that
transcript stream next.
