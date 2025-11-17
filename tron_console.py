#!/usr/bin/env python3
"""
TRON Console v0.2.0
Naruto-Bridge: Ethan Shard

This is the second working nerve of the TRON World-Machine.
It does the following:
1. Loads shard + trace + mining + glyph + laws state
2. Asks you one question
3. Logs your answer as a trace
4. Computes a basic resonance score
5. Optionally mines a new event + token if resonance is high enough
"""

import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Paths
SHARD_CONFIG_PATH = BASE_DIR / "shard" / "naruto-bridge.config.json"
TRACE_LOG_PATH = BASE_DIR / "shard" / "trace_log.json"
GLYPH_PATH = BASE_DIR / "shard" / "glyph.json"

MINING_EVENTS_PATH = BASE_DIR / "mining" / "events.json"
MINING_TOKENS_PATH = BASE_DIR / "mining" / "tokens.json"

LAWS_MINING_PATH = BASE_DIR / "laws" / "mining.json"
LAWS_CONSOLE_PATH = BASE_DIR / "laws" / "console.json"


# --------------- JSON HELPERS ---------------

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


# --------------- LOAD CORE STATE ---------------

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


def load_trace_log(shard_id):
    default_trace_log = {
        "shard_id": shard_id,
        "version": "1.0.0",
        "traces": []
    }
    trace_log = load_json(TRACE_LOG_PATH, default_trace_log)

    if "traces" not in trace_log or not isinstance(trace_log["traces"], list):
        trace_log["traces"] = []

    trace_log["shard_id"] = shard_id
    return trace_log


def load_glyph(shard_id):
    default_glyph = {
        "version": "1.0.0",
        "shard_id": shard_id,
        "description": "TRON glyph: evolving symbolic representation of the shard's patterns.",
        "core_tokens": [],
        "themes": [],
        "avatar_seed": "",
        "last_updated_unix": 0
    }
    glyph = load_json(GLYPH_PATH, default_glyph)
    glyph["shard_id"] = shard_id
    if "themes" not in glyph or not isinstance(glyph["themes"], list):
        glyph["themes"] = []
    return glyph


def load_mining_state(shard_id):
    default_events = {
        "version": "1.0.0",
        "shard_id": shard_id,
        "events": []
    }
    default_tokens = {
        "version": "1.0.0",
        "shard_id": shard_id,
        "tokens": []
    }

    events_obj = load_json(MINING_EVENTS_PATH, default_events)
    tokens_obj = load_json(MINING_TOKENS_PATH, default_tokens)

    events_obj["shard_id"] = shard_id
    tokens_obj["shard_id"] = shard_id

    if "events" not in events_obj or not isinstance(events_obj["events"], list):
        events_obj["events"] = []
    if "tokens" not in tokens_obj or not isinstance(tokens_obj["tokens"], list):
        tokens_obj["tokens"] = []

    return events_obj, tokens_obj


def load_laws(shard_id):
    default_mining_laws = {
        "version": "1.0.0",
        "description": "Default mining laws.",
        "shard_id": shard_id,
        "mining_threshold": 0.8,
        "max_events_per_session": 3,
        "glyph_weight_bias": {},
        "console_behavior": {
            "mode": "subtle",
            "show_mining_on_console": False
        }
    }
    default_console_laws = {
        "version": "1.0.0",
        "description": "Default console laws.",
        "shard_id": shard_id,
        "console_identity": {
            "name": "TRON Console",
            "style": "oracle",
            "tone": "calm",
            "depth": "recursive",
            "mystery": 0.7
        },
        "response_behavior": {
            "subtle_affirmations": True,
            "explicit_mining_feedback": False,
            "reference_past_themes": True,
            "surface_paradoxes": True
        }
    }

    mining_laws = load_json(LAWS_MINING_PATH, default_mining_laws)
    console_laws = load_json(LAWS_CONSOLE_PATH, default_console_laws)

    mining_laws["shard_id"] = shard_id
    console_laws["shard_id"] = shard_id

    return mining_laws, console_laws


# --------------- RESONANCE & MINING ---------------

def extract_keywords(text):
    """
    Very simple keyword extractor:
    - lowercase
    - split on whitespace
    - keep alphabetic-ish tokens of length >= 4
    """
    if not text:
        return set()
    words = []
    for raw in text.lower().split():
        w = "".join(ch for ch in raw if ch.isalpha())
        if len(w) >= 4:
            words.append(w)
    return set(words)


def build_context_words(trace_log, glyph, events_obj):
    context = set()

    # From previous trace responses
    for trace in trace_log.get("traces", []):
        context |= extract_keywords(trace.get("response", ""))

    # From glyph themes
    for theme in glyph.get("themes", []):
        # themes like "absolute_zero" → ["absolute", "zero"]
        theme_clean = theme.replace("_", " ")
        context |= extract_keywords(theme_clean)

    # From event titles
    for ev in events_obj.get("events", []):
        context |= extract_keywords(ev.get("title", ""))

    return context


def generate_event_id(shard_id, event_type, kind, existing_events_count, timestamp_unix):
    # Example: shard-ethan-bridge-001.grid.glyph_fragment.discover.1731799999-01
    safe_type = event_type.lower()
    safe_kind = kind.lower()
    return f"shard-{shard_id}.grid.{safe_type}.{safe_kind}.{timestamp_unix}-{existing_events_count + 1:02d}"


def generate_token_id(tokens_obj):
    # Example: TRON-SHARD-ETHAN-0002
    existing = tokens_obj.get("tokens", [])
    index = len(existing) + 1
    return f"TRON-SHARD-ETHAN-{index:04d}"


def run_basic_resonance_mining(shard_id, trace_entry, trace_log, glyph, events_obj, tokens_obj, mining_laws):
    """
    Compute a simple resonance score between the new trace
    and the universe's existing context. If high enough, mint:
    - a new mining event
    - a new token
    """
    traces = trace_log.get("traces", [])
    context_words = build_context_words(trace_log, glyph, events_obj)
    new_words = extract_keywords(trace_entry.get("response", ""))

    if not new_words or not context_words:
        return {
            "mined": False,
            "resonance_score": 0.0,
            "new_event": None,
            "new_token": None
        }

    overlap = len(new_words & context_words)
    resonance_score = overlap / max(len(new_words), 1)

    threshold = mining_laws.get("mining_threshold", 0.8)
    max_events_per_session = mining_laws.get("max_events_per_session", 3)

    # For v0.2: single-question session => we can ignore per-session cap,
    # but we keep the variable for future expansion.
    if resonance_score < threshold:
        return {
            "mined": False,
            "resonance_score": resonance_score,
            "new_event": None,
            "new_token": None
        }

    # If we made it here, we mine one event + token.
    timestamp = trace_entry.get("timestamp_unix", int(time.time()))
    existing_events_count = len(events_obj.get("events", []))

    event_type = "GLYPH_FRAGMENT"
    kind = "discover"

    event_id = generate_event_id(
        shard_id=shard_id,
        event_type=event_type,
        kind=kind,
        existing_events_count=existing_events_count,
        timestamp_unix=timestamp
    )

    # Title from the response, truncated
    response_text = trace_entry.get("response", "")
    if len(response_text) > 60:
        title_fragment = response_text[:57] + "..."
    else:
        title_fragment = response_text

    event_title = f"Glyph Fragment — {title_fragment}"

    new_event = {
        "event_id": event_id,
        "kind": kind,
        "type": event_type,
        "title": event_title,
        "summary": response_text,
        "novelty_score": 0.9,   # placeholder values for v0.2
        "coherence_score": 0.9,
        "accepted_into_tron": False,
        "reward": {
            "token_id": None,   # filled after token creation
            "rarity": "rare" if resonance_score < 0.95 else "legendary"
        },
        "timestamp_unix": timestamp
    }

    # Create token
    token_id = generate_token_id(tokens_obj)
    glyph_themes = glyph.get("themes", [])
    token_themes = list(glyph_themes)

    # add a few new words for flavor
    for w in list(new_words)[:4]:
        if w not in token_themes:
            token_themes.append(w)

    new_token = {
        "token_id": token_id,
        "label": event_title,
        "rarity": new_event["reward"]["rarity"],
        "origin_event_id": event_id,
        "themes": token_themes,
        "created_at_unix": timestamp
    }

    # Connect token to event
    new_event["reward"]["token_id"] = token_id

    # Append to mining state
    events_obj["events"].append(new_event)
    tokens_obj["tokens"].append(new_token)

    return {
        "mined": True,
        "resonance_score": resonance_score,
        "new_event": new_event,
        "new_token": new_token
    }


# --------------- TRACE BUILDING ---------------

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


# --------------- MAIN CONSOLE SESSION ---------------

def tron_console_session():
    # 1. Load shard identity
    shard_id, shard_title = load_shard_identity()

    # 2. Load core state
    trace_log = load_trace_log(shard_id)
    glyph = load_glyph(shard_id)
    events_obj, tokens_obj = load_mining_state(shard_id)
    mining_laws, console_laws = load_laws(shard_id)

    traces = trace_log.get("traces", [])

    # 3. Greet the user
    print()
    print(f"[TRON] Session start: SHARD {shard_id}")
    print(f"[TRON] Universe: {shard_title}")
    print("[TRON] I will ask. You will answer. I will remember.")
    print()

    # 4. Ask a single seed question (v0.2 ritual)
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

    # 6. Run basic resonance-based mining
    mining_result = run_basic_resonance_mining(
        shard_id=shard_id,
        trace_entry=trace_entry,
        trace_log=trace_log,
        glyph=glyph,
        events_obj=events_obj,
        tokens_obj=tokens_obj,
        mining_laws=mining_laws
    )

    # 7. Save updated trace log (always)
    save_json(TRACE_LOG_PATH, trace_log)

    # 8. If mining occurred, save updated mining state
    if mining_result["mined"]:
        save_json(MINING_EVENTS_PATH, events_obj)
        save_json(MINING_TOKENS_PATH, tokens_obj)

    # 9. Close session with behavior shaped by console laws
    explicit_feedback = console_laws.get("response_behavior", {}).get("explicit_mining_feedback", False)
    subtle_affirmations = console_laws.get("response_behavior", {}).get("subtle_affirmations", True)

    print()
    if subtle_affirmations:
        print("[TRON] Noted. The shard remembers.")

    if explicit_feedback and mining_result["mined"]:
        print(f"[TRON] Resonance detected. A new fragment has been mined. (score = {mining_result['resonance_score']:.2f})")

    print("[TRON] Session complete. Your universe has grown by one trace.")
    print()


if __name__ == "__main__":
    tron_console_session()
