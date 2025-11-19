#!/usr/bin/env python3
"""
TRON Console v0.5.0
Naruto-Bridge: Ethan Shard

Adds:
- Multi-question sessions (loop)
- Session ends when:
  - user types an exit command, OR
  - a token is mined (resonance event)
- Explicit terminal message when a token is mined
- Keeps all previous behavior:
  - procedural prompts
  - mining
  - anomalies
  - stats
  - self-tuning laws
"""

import json
import time
import random
from pathlib import Path

from core.random_engine import RandomizationEngine

BASE_DIR = Path(__file__).resolve().parent

# Paths
SHARD_CONFIG_PATH = BASE_DIR / "shard" / "naruto-bridge.config.json"
TRACE_LOG_PATH = BASE_DIR / "shard" / "trace_log.json"
GLYPH_PATH = BASE_DIR / "shard" / "glyph.json"

MINING_EVENTS_PATH = BASE_DIR / "mining" / "events.json"
MINING_TOKENS_PATH = BASE_DIR / "mining" / "tokens.json"

LAWS_MINING_PATH = BASE_DIR / "laws" / "mining.json"
LAWS_CONSOLE_PATH = BASE_DIR / "laws" / "console.json"

ANOMALIES_INDEX_PATH = BASE_DIR / "anomalies" / "index.json"

STATS_PATH = BASE_DIR / "stats" / "world_stats.json"
WORLD_STATE_PATH = BASE_DIR / "world" / "world_state.json"

# LLM toggle placeholder (future integration)
USE_LLM = False


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


def load_anomalies(shard_id):
    default_anomalies = {
        "version": "1.0.0",
        "description": "Registry of paradoxes, contradictions, and emergent anomalies detected by the TRON World-Machine.",
        "shard_id": shard_id,
        "anomalies": []
    }
    anomalies = load_json(ANOMALIES_INDEX_PATH, default_anomalies)
    anomalies["shard_id"] = shard_id
    if "anomalies" not in anomalies or not isinstance(anomalies["anomalies"], list):
        anomalies["anomalies"] = []
    return anomalies


def load_stats(shard_id):
    default_stats = {
        "version": "1.0.0",
        "shard_id": shard_id,
        "total_sessions": 0,
        "total_traces": 0,
        "total_mined_events": 0,
        "total_anomalies": 0,
        "avg_answer_length": 0.0
    }
    stats = load_json(STATS_PATH, default_stats)
    stats["shard_id"] = shard_id
    return stats


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
        theme_clean = theme.replace("_", " ")
        context |= extract_keywords(theme_clean)

    # From event titles
    for ev in events_obj.get("events", []):
        context |= extract_keywords(ev.get("title", ""))

    return context


def generate_event_id(shard_id, event_type, kind, existing_events_count, timestamp_unix):
    safe_type = event_type.lower()
    safe_kind = kind.lower()
    return f"shard-{shard_id}.grid.{safe_type}.{safe_kind}.{timestamp_unix}-{existing_events_count + 1:02d}"


def generate_token_id(tokens_obj):
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
        "novelty_score": 0.9,
        "coherence_score": 0.9,
        "accepted_into_tron": False,
        "reward": {
            "token_id": None,
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


# --------------- ANOMALY DETECTION ---------------

COLD_WORDS = {"cold", "frozen", "freeze", "ice", "icy", "snow", "zero", "void"}
HOT_WORDS = {"heat", "hot", "fire", "flame", "burn", "burning", "ember", "lava"}


def generate_anomaly_id(anomalies_obj):
    existing = anomalies_obj.get("anomalies", [])
    index = len(existing) + 1
    return f"anomaly-{index:04d}"


def detect_loop_anomaly(trace_log, trace_entry):
    """
    Detect exact repetition of a previous response.
    """
    new_resp = (trace_entry.get("response") or "").strip()
    if not new_resp:
        return None

    for prev in trace_log.get("traces", [])[:-1]:
        prev_resp = (prev.get("response") or "").strip()
        if prev_resp and prev_resp == new_resp:
            return {
                "kind": "exact_loop",
                "description": "The shard has spoken the same phrase again.",
                "related_trace_ids": [prev.get("trace_id"), trace_entry.get("trace_id")]
            }

    return None


def detect_polarity_anomaly(trace_log, glyph, trace_entry):
    """
    Very simple polarity detection:
    - If glyph/themes/context are strongly cold-coded
      and new response uses hot words, or vice versa.
    """
    response_text = (trace_entry.get("response") or "").lower()
    if not response_text:
        return None

    new_words = extract_keywords(response_text)

    # Build a simple cold/hot bias from glyph + past context
    context_words = set()
    for t in trace_log.get("traces", []):
        context_words |= extract_keywords(t.get("response", ""))

    for theme in glyph.get("themes", []):
        theme_clean = theme.replace("_", " ")
        context_words |= extract_keywords(theme_clean)

    cold_bias = len(context_words & COLD_WORDS)
    hot_bias = len(context_words & HOT_WORDS)

    new_cold = len(new_words & COLD_WORDS)
    new_hot = len(new_words & HOT_WORDS)

    # If universe is cold-biased and response is hot-heavy
    if cold_bias > hot_bias and new_hot > 0 and new_cold == 0:
        return {
            "kind": "polarity_tension",
            "description": "Hot imagery appears in a shard long steeped in cold themes.",
            "related_trace_ids": [trace_entry.get("trace_id")]
        }

    # If universe is hot-biased and response is cold-heavy
    if hot_bias > cold_bias and new_cold > 0 and new_hot == 0:
        return {
            "kind": "polarity_tension",
            "description": "Cold imagery appears in a shard long steeped in heat and fire.",
            "related_trace_ids": [trace_entry.get("trace_id")]
        }

    return None


def run_anomaly_detection(shard_id, trace_log, glyph, trace_entry, anomalies_obj):
    anomalies_found = []

    loop_result = detect_loop_anomaly(trace_log, trace_entry)
    if loop_result:
        anomalies_found.append(loop_result)

    polarity_result = detect_polarity_anomaly(trace_log, glyph, trace_entry)
    if polarity_result:
        anomalies_found.append(polarity_result)

    if not anomalies_found:
        return {
            "spawned": False,
            "new_anomalies": []
        }

    timestamp = trace_entry.get("timestamp_unix", int(time.time()))
    new_records = []

    for a in anomalies_found:
        anomaly_id = generate_anomaly_id(anomalies_obj)
        record = {
            "anomaly_id": anomaly_id,
            "kind": a["kind"],
            "description": a["description"],
            "related_trace_ids": a.get("related_trace_ids", []),
            "timestamp_unix": timestamp,
            "severity": "medium"
        }
        anomalies_obj["anomalies"].append(record)
        new_records.append(record)

    return {
        "spawned": True,
        "new_anomalies": new_records
    }


# --------------- STATS & SELF-TUNING ---------------

def update_stats_for_trace(stats, trace_entry, mining_result, anomaly_result):
    stats["total_traces"] += 1

    answer_len = len((trace_entry.get("response") or ""))
    n = stats["total_traces"]
    prev_avg = stats.get("avg_answer_length", 0.0)
    stats["avg_answer_length"] = prev_avg + (answer_len - prev_avg) / max(n, 1)

    if mining_result.get("mined"):
        stats["total_mined_events"] += 1

    if anomaly_result.get("spawned"):
        stats["total_anomalies"] += len(anomaly_result.get("new_anomalies", []))

    return stats


def self_tune_laws(mining_laws, console_laws, stats):
    sessions = stats.get("total_sessions", 0)
    mined = stats.get("total_mined_events", 0)

    if sessions == 0:
        return mining_laws, console_laws

    mining_rate = mined / sessions
    threshold = mining_laws.get("mining_threshold", 0.8)

    target_low = 0.2
    target_high = 0.7
    delta = 0.02

    if mining_rate < target_low:
        threshold = max(0.4, threshold - delta)
    elif mining_rate > target_high:
        threshold = min(0.95, threshold + delta)

    mining_laws["mining_threshold"] = threshold

    avg_len = stats.get("avg_answer_length", 0.0)
    identity = console_laws.get("console_identity", {})
    mystery = identity.get("mystery", 0.7)

    if avg_len < 50:
        mystery = max(0.3, mystery - 0.02)
    else:
        mystery = min(0.95, mystery + 0.02)

    identity["mystery"] = mystery
    console_laws["console_identity"] = identity

    return mining_laws, console_laws


# --------------- PROMPT GENERATION ---------------

WH_WORDS = ["what", "why", "how", "when"]
VERBS = ["see", "feel", "remember", "fear", "want", "carry", "avoid", "hear"]
NOUNS = ["image", "memory", "feeling", "place", "sound", "pattern", "dream"]
FRAMES = [
    "lately",
    "today",
    "from your past",
    "when you are alone",
    "when you think about the future"
]


def simple_random_prompt():
    wh = random.choice(WH_WORDS)
    verb = random.choice(VERBS)
    noun = random.choice(NOUNS)
    frame = random.choice(FRAMES)
    return f"{wh.capitalize()} {noun} do you {verb} {frame}?"


def get_hot_words(glyph, events_obj, limit=8):
    hot = set()

    for theme in glyph.get("themes", []):
        hot.update(theme.replace("_", " ").split())

    for ev in events_obj.get("events", []):
        hot.update(ev.get("title", "").lower().split())

    clean = []
    for w in hot:
        w2 = "".join(ch for ch in w if ch.isalpha())
        if len(w2) >= 4:
            clean.append(w2)

    random.shuffle(clean)
    return clean[:limit]


def generate_prompt(trace_log, glyph, events_obj):
    base_prompt = simple_random_prompt()
    hot_words = get_hot_words(glyph, events_obj, limit=4)

    if not hot_words:
        return base_prompt

    if random.random() < 0.5:
        hw = random.choice(hot_words)
        return base_prompt[:-1] + f" about {hw}?"

    return base_prompt


def generate_prompt_with_state(trace_log, glyph, events_obj):
    if USE_LLM:
        # Future integration: build state summary, call LLM
        return generate_prompt(trace_log, glyph, events_obj)
    else:
        return generate_prompt(trace_log, glyph, events_obj)


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
    shard_id, shard_title = load_shard_identity()

    trace_log = load_trace_log(shard_id)
    glyph = load_glyph(shard_id)
    events_obj, tokens_obj = load_mining_state(shard_id)
    mining_laws, console_laws = load_laws(shard_id)
    anomalies_obj = load_anomalies(shard_id)
    stats = load_stats(shard_id)
    engine = RandomizationEngine()
    engine.load_state(WORLD_STATE_PATH)

    traces = trace_log.get("traces", [])

    stats["total_sessions"] += 1

    print()
    print(f"[TRON] Session start: SHARD {shard_id}")
    print(f"[TRON] Universe: {shard_title}")
    print("[TRON] I will ask. You will answer. I will remember.")
    print("[TRON] Type 'exit', 'quit', or ':q' to end the session.")
    print()

    session_should_end = False

    while not session_should_end:
        prompt_text = generate_prompt_with_state(trace_log, glyph, events_obj)
        print(f"[TRON] {prompt_text}")
        user_response = input("> ").strip()

        if not user_response:
            print("[TRON] Empty response. Session closed.")
            break

        lower_resp = user_response.lower()
        if lower_resp in ("exit", "quit", ":q"):
            print("[TRON] Understood. Closing session at your request.")
            break

        trace_entry = build_trace_entry(
            shard_id=shard_id,
            prompt_text=prompt_text,
            user_response=user_response,
            existing_traces_count=len(traces)
        )
        traces.append(trace_entry)
        trace_log["traces"] = traces
        trace_log["shard_id"] = shard_id

        mining_result = run_basic_resonance_mining(
            shard_id=shard_id,
            trace_entry=trace_entry,
            trace_log=trace_log,
            glyph=glyph,
            events_obj=events_obj,
            tokens_obj=tokens_obj,
            mining_laws=mining_laws
        )

        if mining_result["mined"]:
            new_token = mining_result.get("new_token")
            if new_token:
                engine.register_mined_token(new_token)

        anomaly_result = run_anomaly_detection(
            shard_id=shard_id,
            trace_log=trace_log,
            glyph=glyph,
            trace_entry=trace_entry,
            anomalies_obj=anomalies_obj
        )

        stats = update_stats_for_trace(stats, trace_entry, mining_result, anomaly_result)
        mining_laws, console_laws = self_tune_laws(mining_laws, console_laws, stats)

        save_json(TRACE_LOG_PATH, trace_log)
        save_json(STATS_PATH, stats)

        if mining_result["mined"]:
            save_json(MINING_EVENTS_PATH, events_obj)
            save_json(MINING_TOKENS_PATH, tokens_obj)

        if anomaly_result["spawned"]:
            save_json(ANOMALIES_INDEX_PATH, anomalies_obj)

        save_json(LAWS_MINING_PATH, mining_laws)
        save_json(LAWS_CONSOLE_PATH, console_laws)

        # --- World exploration step ---
        units = engine.travel_units(4)

        current_zone = engine.get_current_zone()
        current_region = engine.get_current_region()

        if current_zone:
            zone_name = current_zone.name or f"Zone {current_zone.id}"
            print(f"[TRON] The shard moves through a new zone: {zone_name}")
        if current_region:
            region_name = current_region.name or f"Region {current_region.id}"
            print(f"[TRON] The shard now resides within region: {region_name}")

        engine.save_state(WORLD_STATE_PATH)

        response_behavior = console_laws.get("response_behavior", {})
        subtle_affirmations = response_behavior.get("subtle_affirmations", True)
        explicit_mining_feedback = response_behavior.get("explicit_mining_feedback", False)
        surface_paradoxes = response_behavior.get("surface_paradoxes", True)

        print()
        if subtle_affirmations:
            print("[TRON] Noted. The shard remembers.")

        if mining_result["mined"]:
            label = mining_result["new_event"]["title"] if mining_result["new_event"] else "a new fragment"
            print(f"[TRON] A fragment has crystallized into a token: {label}")
            session_should_end = True

            if explicit_mining_feedback:
                print(f"[TRON] Resonance score: {mining_result['resonance_score']:.2f}")

        if surface_paradoxes and anomaly_result["spawned"]:
            print("[TRON] Something in that answer does not fully reconcile. The shard will think on it.")

        print()

    print("[TRON] Session complete. Your universe has grown.")
    print()


if __name__ == "__main__":
    tron_console_session()
