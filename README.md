# Streaming Voice Agent

A real-time voice agent you can hold a spoken conversation with over a live WebSocket connection — you talk, it detects when you've actually stopped (not just paused), transcribes your turn, generates a reply, and speaks it back, entirely on CPU with no GPU involved.


<img width="821" height="482" alt="Screenshot 2026-08-25 220841" src="https://github.com/user-attachments/assets/c7f1ddfd-cec7-40f9-b5bc-d219caaeda22" />



## How it works

The pipeline runs as a WebSocket server that a client streams raw microphone audio to, continuously, in small chunks — not as one uploaded file. Each turn goes through five stages:

1. **Streaming audio in** — the client sends PCM16 audio over the WebSocket as it's recorded.
2. **Turn detection (VAD)** — Silero VAD scores each audio window for speech, and a small state machine on top declares a turn "over" after ~600ms of continuous silence.
3. **Speech-to-text** — the buffered utterance is transcribed in one shot with `faster-whisper` (`base` model, int8), a CPU-optimized speech recognizer.
4. **Reply generation** — the transcript is sent to Gemini, which streams its response token by token. Time-to-first-token (TTFT) is measured separately from total generation time, since TTFT is what a user actually perceives as "how long until it starts responding."
5. **Text-to-speech** — the reply text is converted to audio with `pyttsx3`, an offline, CPU-only TTS engine.

The reply audio is sent back over the same connection, and every stage's timing is logged to `logs/latency_log.csv` — VAD, STT, LLM TTFT, LLM total, TTS, and the end-to-end total per turn.

## Architecture

```
voice-agent/
├── server.py       # WebSocket server — orchestrates the pipeline, per-stage timing
├── client.py        # Test harness — streams mic audio to the server, plays back replies
├── vad.py            # Silero VAD wrapper — turn/endpoint detection
├── stt.py             # faster-whisper wrapper — audio → text
├── llm.py              # Gemini wrapper — text → reply, streamed for TTFT
├── tts.py               # pyttsx3 wrapper — text → audio
├── latency.py             # CSV logging helper
├── requirements.txt
├── template.sh
├── .env                     # GEMINI_API_KEY (not committed)
└── logs/
    └── latency_log.csv        # per-turn latency table
```

## Setup

```bash
chmod +x template.sh
./template.sh
```

This creates a virtual environment, installs dependencies, and generates a `.env` file for your Gemini API key (get one at https://aistudio.google.com/apikey).

## Running it

Two terminals, both with the virtual environment activated:

```bash
# Terminal 1 — start the server
python server.py

# Terminal 2 — start talking
python client.py
```

Speak into your mic, pause for about half a second, and the agent will reply out loud.

## Latency results

Findings from test runs, per turn:

| Stage | Typical time |
|---|---|
| VAD | < 1 ms |
| STT (faster-whisper, int8) | ~0.5–1.8 s |
| LLM TTFT (Gemini) | ~2–3.6 s |
| LLM total | ~2–3.6 s |
| TTS (pyttsx3) | ~0.1–1.1 s |
| **Total round-trip** | **~4–5 s** |

VAD and TTS stayed well under a second combined across every test run. Gemini's time-to-first-token was consistently the largest chunk of the total — pointing to starting TTS on partial replies instead of waiting for the full response as the next optimization.

## Tech stack

- **Transport:** `websockets` (Python), raw PCM16 audio frames
- **VAD / turn detection:** Silero VAD
- **STT:** `faster-whisper` (CTranslate2 backend, int8 quantization)
- **LLM:** Gemini API (streaming)
- **TTS:** `pyttsx3` (offline, CPU-based)

## Possible next steps

- Stream TTS on partial LLM output instead of waiting for the full reply, to cut perceived latency
- Swap batch STT for a streaming/incremental transcription approach
- Replace `pyttsx3` with a higher-quality CPU TTS model (e.g., an ONNX-exported model) while keeping it offline
