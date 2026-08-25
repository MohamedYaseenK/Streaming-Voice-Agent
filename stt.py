"""
Speech-to-text on a finalized utterance. This is NOT streaming STT --
we only transcribe once, after VAD has told us the turn is over.
That's a deliberate simplification: streaming/partial transcription
is a real feature but adds a lot of complexity for a v1 demo.
"""
import numpy as np
from faster_whisper import WhisperModel

MODEL_SIZE = "base"  # CPU-friendly. Try "small" for more accuracy if latency allows.

_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


def transcribe(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """audio: float32 mono numpy array in [-1, 1]."""
    segments, _ = _model.transcribe(audio, language="en", beam_size=1)
    return " ".join(seg.text.strip() for seg in segments).strip()
