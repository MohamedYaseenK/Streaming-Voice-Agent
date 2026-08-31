voice-agent/
├── server.py     # the pipeline — everything lives here
├── client.py     # test harness: mic → websocket → speaker
├── vad.py        # endpoint detection (Silero)
├── stt.py        # faster-whisper
├── llm.py        # Claude, streamed for TTFT
├── tts.py        # pyttsx3
├── latency.py    # CSV logger
├── requirements.txt
└── template.sh