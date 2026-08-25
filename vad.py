"""
Endpoint detection: figures out when the user has STOPPED talking
(not just that they're talking). Silero VAD's streaming iterator
already implements the start/end state machine we need -- we just
wrap it in a small class with a clean interface for server.py.
"""
from silero_vad import load_silero_vad, VADIterator
import numpy as np

SAMPLE_RATE = 16000
WINDOW_SIZE = 512  # samples per VAD call -- fixed requirement for 16kHz audio


class EndpointDetector:
    def __init__(self, min_silence_ms: int = 600, threshold: float = 0.5):
        """
        min_silence_ms: how long the user must be silent before we
        declare their turn "ended". Lower = snappier but more likely
        to cut someone off mid-sentence. 500-700ms is a reasonable start.
        """
        self.model = load_silero_vad()
        self.iterator = VADIterator(
            self.model,
            sampling_rate=SAMPLE_RATE,
            threshold=threshold,
            min_silence_duration_ms=min_silence_ms,
        )
        self.speaking = False

    def process_chunk(self, chunk: np.ndarray):
        """
        chunk: float32 numpy array, exactly WINDOW_SIZE samples, range [-1, 1].
        Returns "start", "end", or None.
        """
        event = self.iterator(chunk, return_seconds=False)
        if event is None:
            return None
        if "start" in event:
            self.speaking = True
            return "start"
        if "end" in event:
            self.speaking = False
            return "end"
        return None

    def reset(self):
        self.iterator.reset_states()
        self.speaking = False
