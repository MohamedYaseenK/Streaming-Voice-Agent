"""
Turns a transcript into a spoken-style response, using Gemini.
Streams the completion so we can measure time-to-first-token (TTFT) --
for a voice agent, TTFT matters more to perceived latency than total
generation time, since TTS could start as soon as the first sentence
is out (a further optimization this v1 doesn't implement yet).
"""
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

MODEL_NAME = "gemini-3.6-flash"  

SYSTEM_PROMPT = (
    "You are a voice assistant. Keep replies to 1-2 short sentences -- "
    "they will be spoken aloud, not read."
)

_model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT)


def respond(user_text: str, history: list) -> tuple[str, float]:
    """
    Returns (full_response_text, ttft_seconds).
    `history` is a list of {"role": "user"/"model", "parts": [text]} dicts
    (Gemini's chat format), mutated in place so the conversation carries
    context across turns.
    """
    history.append({"role": "user", "parts": [user_text]})
    start = time.perf_counter()
    ttft = None
    chunks = []

    stream = _model.generate_content(history, stream=True)
    for chunk in stream:
        if chunk.text:
            if ttft is None:
                ttft = time.perf_counter() - start
            chunks.append(chunk.text)

    full_text = "".join(chunks)
    history.append({"role": "model", "parts": [full_text]})
    return full_text, (ttft or 0.0)
