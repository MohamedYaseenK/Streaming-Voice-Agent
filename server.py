"""
WebSocket server -- the whole pipeline lives here.

Protocol (kept intentionally dumb, no framing library needed):
  - Client sends BINARY frames: raw PCM16LE mono audio @ 16kHz, any chunk size.
  - Server buffers + VADs it internally. When it detects end-of-turn:
      1. sends back one TEXT frame: JSON with transcript, reply, and
         per-stage latencies
      2. sends back one BINARY frame: the response audio (WAV bytes)
  - Then keeps listening for the next turn, same connection.
"""
import asyncio
import json
import time

import numpy as np
import websockets

from vad import EndpointDetector, WINDOW_SIZE
from stt import transcribe
from llm import respond
from tts import synthesize
from latency import log_row

BYTES_PER_SAMPLE = 2  # int16


async def handle_connection(ws):
    print("client connected")
    detector = EndpointDetector(min_silence_ms=600)
    utterance_buf = []   # float32 chunks collected while the user is speaking
    pcm_leftover = b""   # bytes that don't yet fill a full VAD window
    history = []          # running conversation, for multi-turn context

    async for message in ws:
        if not isinstance(message, (bytes, bytearray)):
            continue  # ignore stray non-audio frames

        pcm_leftover += message
        window_bytes = WINDOW_SIZE * BYTES_PER_SAMPLE

        while len(pcm_leftover) >= window_bytes:
            frame_bytes = pcm_leftover[:window_bytes]
            pcm_leftover = pcm_leftover[window_bytes:]

            int16 = np.frombuffer(frame_bytes, dtype=np.int16)
            float32 = int16.astype(np.float32) / 32768.0

            vad_start = time.perf_counter()
            event = detector.process_chunk(float32)
            vad_ms = (time.perf_counter() - vad_start) * 1000

            if detector.speaking or event == "end":
                utterance_buf.append(float32)

            if event == "end" and utterance_buf:
                audio = np.concatenate(utterance_buf)
                utterance_buf = []
                asyncio.create_task(run_pipeline(ws, audio, history, vad_ms))


async def run_pipeline(ws, audio: np.ndarray, history: list, vad_ms: float):
    t0 = time.perf_counter()

    t = time.perf_counter()
    transcript = transcribe(audio)
    stt_ms = (time.perf_counter() - t) * 1000

    if not transcript:
        return  # noise / silence picked up as a false endpoint -- skip it

    t = time.perf_counter()
    reply_text, ttft = respond(transcript, history)
    llm_total_ms = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    audio_bytes = synthesize(reply_text)
    tts_ms = (time.perf_counter() - t) * 1000

    total_ms = (time.perf_counter() - t0) * 1000

    latency = {
        "vad": round(vad_ms, 1),
        "stt": round(stt_ms, 1),
        "llm_ttft": round(ttft * 1000, 1),
        "llm_total": round(llm_total_ms, 1),
        "tts": round(tts_ms, 1),
        "total": round(total_ms, 1),
    }
    payload = {"transcript": transcript, "reply": reply_text, "latency_ms": latency}
    print(payload)

    await ws.send(json.dumps(payload))
    await ws.send(audio_bytes)

    log_row({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vad_ms": latency["vad"],
        "stt_ms": latency["stt"],
        "llm_ttft_ms": latency["llm_ttft"],
        "llm_total_ms": latency["llm_total"],
        "tts_ms": latency["tts"],
        "total_ms": latency["total"],
    })


async def main():
    async with websockets.serve(handle_connection, "0.0.0.0", 8765, max_size=None):
        print("voice-agent server listening on ws://0.0.0.0:8765")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
