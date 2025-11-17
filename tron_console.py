#!/usr/bin/env python3
"""
TRON Console v0.1.0
Naruto-Bridge: Ethan Shard

This is the first working nerve of the TRON World-Machine.
It does exactly three things:
1. Loads shard + trace state
2. Asks you one question
3. Logs your answer as a trace

No mining, no anomalies yet. Just memory.
"""

import json
import os
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

SHARD_CONFIG_PATH = BASE_DIR / "shard" / "naruto-bridge.config.json"
TRACE_LOG_PATH = BASE_DIR / "shard" / "trace_log.json"


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        print(f"[TRON] WARNING: JSON decode error in {path}. Using default in-memory structure.")
        return default


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_shard_identity():
    default_config = {
        "shard": {
            "shard_id": "unknown-shard",
            "universe_manifest": {
                "title": "Unnamed Shard"
            }
        }
    }
    config = load_json(SHARD_CONFIG_PATH, default_config)

    shard = config.get("shard", {})
    shard_id = shard.get("shard_id", "unknown-shard")

    manifest = shard.get("universe_manifest", {})
    title = manifest.get("title", "Unnamed Shard")

    return shard_id, title


def load_trace_log():
    default_trace_log = {
        "shard_id": "unknown-shard",
        "version": "1.0.0",
        "traces": []
    }
    trace_log = load_json(TRACE_LOG_PATH, default_trace_log)

    # Ensure keys exist
    if "traces" not in trace_log or not isinstance(trace_log["traces"], list):
        trace_log["traces"] = []

    return trace_log


def build_trace_entry(shard_id, prompt_text, user_response, existing_traces_count):
    timestamp = int(time.time())
    trace_id = f"trace-{existing_traces_count + 1:04d}"

    return {
        "trace_id": trace_id,
        "timestamp_unix": timestamp,
        "shard_id": shard_id,
        "source": "tron_console",
        "prompt": prompt_text,
        "response": user_response
    }


def tron_console_session():
    # 1. Load shard identity
    shard_id, shard_title = load_shard_identity()

    # 2. Load trace log
    trace_log = load_trace_log()
    traces = trace_log.get("traces", [])

    # 3. Greet the user
    print()
    print(f"[TRON] Session start: SHARD {shard_id}")
    print(f"[TRON] Universe: {shard_title}")
    print("[TRON] I will ask. You will answer. I will remember.")
    print()

    # 4. Ask a single seed question (v0.1 ritual)
    prompt_text = "Q1: What image or feeling has been following you around lately?"
    print(f"[TRON] {prompt_text}")
    user_response = input("> ").strip()

    if not user_response:
        print("[TRON] No response received. Session closed without trace.")
        return

    # 5. Build and append trace entry
    trace_entry = build_trace_entry(
        shard_id=shard_id,
        prompt_text=prompt_text,
        user_response=user_response,
        existing_traces_count=len(traces)
    )
    traces.append(trace_entry)
    trace_log["traces"] = traces

    # Ensure shard_id is set in trace log
    trace_log["shard_id"] = shard_id

    # 6. Save updated trace log
    save_json(TRACE_LOG_PATH, trace_log)

    # 7. Close session
    print()
    print("[TRON] Noted. The shard remembers.")
    print("[TRON] Session complete. Your universe has grown by one trace.")
    print()


if __name__ == "__main__":
    tron_console_session()
