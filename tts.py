"""
Synthesizes the response text back to audio. pyttsx3 is offline and
CPU-only (good for our constraint) but it can't hand us raw bytes
directly -- so we write to a temp WAV file and read it back. Fine
for a demo; a production system would stream TTS chunks instead.
"""
import os
import tempfile
import pyttsx3


def synthesize(text: str) -> bytes:
    engine = pyttsx3.init()
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        engine.save_to_file(text, path)
        engine.runAndWait()
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)
