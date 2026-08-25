"""
Small CSV logging helper. This produces the deliverable that matters
most: a per-turn, stage-by-stage latency table.
"""
import csv
import os
import time

LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "latency_log.csv")
FIELDS = ["timestamp", "vad_ms", "stt_ms", "llm_ttft_ms", "llm_total_ms", "tts_ms", "total_ms"]


def log_row(row: dict):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
