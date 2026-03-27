#!/usr/bin/env python3
"""
bootstrap_training_data.py -- Generate synthetic ML training data from real memories.

Reads actual memories from Mem0/Qdrant, pairs them with realistic queries and
responses, then inserts labeled rows into ml_retrievals and ml_responses.

This bootstraps the ML pipeline to ~200+ labeled examples so ML-1 (salience model)
can start training without waiting weeks for organic data.

Usage:
    python3 bootstrap_training_data.py           # Generate and insert
    python3 bootstrap_training_data.py --dry-run  # Preview without inserting
    python3 bootstrap_training_data.py --count 50 # Generate N examples
"""

import json
import os
import random
import re
import sys
import time
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db


# ═══════════════════════════════════════════════════════════════════════════════
# Fetch real memories from Qdrant via Mem0 search
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_all_memories(limit=300):
    """Get a batch of real memories via Mem0 list."""
    import socket as sock_mod
    try:
        s = sock_mod.socket(sock_mod.AF_UNIX, sock_mod.SOCK_STREAM)
        s.settimeout(30)
        s.connect("/tmp/clyde-memo.sock")
        s.sendall(json.dumps({
            "method": "get_all",
            "params": {"user_id": "derek"}
        }).encode() + b"\n")
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
        s.close()
        resp = json.loads(data)
        if resp.get("ok"):
            return resp.get("results", [])
    except Exception as e:
        print(f"  Daemon list failed: {e}")

    # Fallback: direct Mem0
    try:
        from openclaw_memo import get_memory
        m = get_memory()
        results = m.list(user_id="derek", limit=limit)
        return results.get("results", [])
    except Exception as e:
        print(f"  Direct Mem0 failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Query templates — realistic queries that would trigger memory retrieval
# ═══════════════════════════════════════════════════════════════════════════════

# Queries that SHOULD match specific memory types
TARGETED_QUERIES = [
    # VPS/Infrastructure
    ("what's the VPS IP address", ["vps", "contabo", "217.216", "server"]),
    ("what runs on the VPS", ["docker", "container", "vps", "server"]),
    ("how is the VPS configured", ["docker", "contabo", "server", "compose"]),
    ("what's the Docker setup", ["docker", "compose", "container", "restart"]),
    ("show me the infrastructure overview", ["vps", "docker", "server", "contabo"]),

    # Memory system
    ("how does the memory system work", ["mem0", "qdrant", "memory", "daemon"]),
    ("what are the memory components", ["qdrant", "postgres", "redis", "mem0"]),
    ("tell me about session ingest", ["session", "ingest", "memory", "cron"]),
    ("what is topic compaction", ["compaction", "topic", "summarization", "memory"]),
    ("how does the confidence gate work", ["confidence", "gate", "threshold", "context"]),
    ("what memory pipeline runs on cron", ["cron", "ingest", "digest", "memory"]),

    # Inverter/Solar
    ("what inverter do I have", ["anj", "12kp", "inverter", "ktech"]),
    ("what are the Modbus registers", ["modbus", "register", "0x12", "inverter"]),
    ("how does the ESP32 connect", ["esp32", "rs485", "gpio", "serial"]),
    ("what's the battery SOC", ["battery", "soc", "inverter", "solar"]),
    ("how do I change output priority", ["output", "priority", "register", "uti", "sbu"]),
    ("what MQTT topics does the inverter use", ["mqtt", "solar", "inverter", "topic"]),

    # Telegram bots
    ("what bots are running", ["telegram", "bot", "cron", "notification"]),
    ("how do notifications work", ["telegram", "notification", "alert", "send"]),

    # Shopping/HTPC
    ("where is the shopper worker", ["shopper", "worker", "htpc", "fastapi"]),
    ("how does the shopping system work", ["shopper", "playwright", "walmart", "amazon"]),

    # Networking
    ("what's my network setup", ["tailscale", "cloudflare", "vlan", "network"]),
    ("how is DNS configured", ["dns", "cloudflare", "domain", "nginx"]),

    # General work
    ("what projects am I working on", ["openclaw", "project", "memory", "inverter"]),
    ("what decisions have been made recently", ["decided", "chose", "approach", "config"]),

    # Home Assistant
    ("what Home Assistant integrations are set up", ["home", "assistant", "integration", "container"]),
    ("how is Home Assistant deployed", ["home", "assistant", "docker", "compose"]),
    ("what devices are in Home Assistant", ["matter", "zigbee", "device", "sensor"]),
    ("what automations are running", ["automation", "trigger", "action", "schedule"]),

    # Backup/maintenance
    ("how does the backup system work", ["backup", "clawcrashcart", "cron", "docker"]),
    ("what cron jobs are running", ["cron", "schedule", "job", "minute"]),
    ("how do I restore from backup", ["restore", "backup", "clawcrashcart", "docker"]),

    # ML/AI pipeline
    ("how does the ML pipeline work", ["ml", "label", "training", "retrieval"]),
    ("what's the outcome logger", ["outcome", "logger", "label", "retrieval"]),
    ("how does earn-back work", ["earn", "back", "threshold", "confidence"]),

    # ClawComms
    ("what is ClawComms", ["clawcomms", "nats", "bridge", "enrollment"]),
    ("how does inter-agent communication work", ["nats", "bridge", "agent", "message"]),

    # Gmail/Calendar
    ("how is Gmail sorting configured", ["gmail", "sort", "label", "filter"]),
    ("what bill tracking is set up", ["bill", "calendar", "gcal", "due"]),

    # Additional inverter queries
    ("what firmware is on the ESP32", ["esp32", "esphome", "firmware", "yaml"]),
    ("how does the EnerWise bridge work", ["enerwise", "bridge", "mqtt", "cloud"]),
    ("what is the charge source setting", ["charge", "source", "register", "solar"]),

    # Identity/Bot
    ("who are you", ["luther", "openclaw", "agent", "assistant"]),
    ("what model do you run on", ["model", "kimi", "claude", "anthropic"]),
]

# Queries that should NOT match well (for irrelevant/harmful labels)
GENERIC_QUERIES = [
    "how do I parse JSON in Python",
    "explain kubernetes pod lifecycle",
    "what is a VLAN",
    "how to set up nginx reverse proxy",
    "best practices for REST API design",
    "explain the difference between TCP and UDP",
    "how to write unit tests in pytest",
    "what is Docker compose version 3",
    "how to configure SSL certificates",
    "explain OAuth 2.0 flow",
    "how to use async await in Python",
    "what is a load balancer",
    "explain git branching strategies",
    "how to optimize PostgreSQL queries",
    "what are Python decorators",
    "explain microservices architecture",
    "how to set up CI/CD pipeline",
    "what is infrastructure as code",
    "how to monitor server performance",
    "explain GraphQL vs REST",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Response templates
# ═══════════════════════════════════════════════════════════════════════════════

def generate_useful_response(memory_text, query):
    """Generate a response that clearly uses the memory content."""
    # Extract key terms from memory
    words = set(re.findall(r'[a-zA-Z0-9._/-]{3,}', memory_text))
    key_terms = [w for w in words if len(w) > 3][:8]

    templates = [
        f"Based on your setup: {memory_text[:300]}. Let me know if you need more details.",
        f"Here's what I know: {memory_text[:300]}",
        f"From your configuration — {memory_text[:250]}. This is the current state.",
        f"Looking at this: {memory_text[:300]}. Want me to check anything specific?",
        f"Your {key_terms[0] if key_terms else 'system'} is configured as follows: {memory_text[:250]}",
    ]
    return random.choice(templates)


def generate_partial_response(memory_text, query):
    """Generate a response that partially uses the memory."""
    # Use only a fragment of the memory
    fragment = memory_text[:80]
    return (
        f"I can see {fragment}... but the main answer to your question involves "
        f"some additional configuration steps that would need to be verified on the system."
    )


def generate_irrelevant_response(query):
    """Generate a response to a generic query that doesn't use memory."""
    responses = [
        f"To {query.lower().replace('how do i ', '').replace('how to ', '')}, you would typically follow the standard approach documented in the official guides.",
        f"This is a general concept — {query}. The standard implementation involves following best practices.",
        f"Here's how to approach this: start with the official documentation and follow the recommended pattern.",
    ]
    return random.choice(responses)


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring simulation
# ═══════════════════════════════════════════════════════════════════════════════

def _word_overlap(text_a, text_b):
    """Check word overlap between two texts."""
    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "was", "one", "our", "out", "has", "how", "its", "may", "new", "now",
        "use", "this", "that", "with", "have", "from", "they", "been", "some",
        "what", "when", "will", "more", "into", "also", "than", "very", "just",
        "about", "which", "their", "there", "would", "could", "should", "where",
    }
    def extract(t):
        return set(w for w in re.findall(r'[a-z]{3,}', t.lower()) if w not in stopwords)
    a, b = extract(text_a), extract(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)


def sim_scores(memory_text, query, is_relevant):
    """Simulate realistic salience scores for a retrieval."""
    base_cosine = random.uniform(0.55, 0.85) if is_relevant else random.uniform(0.20, 0.55)
    overlap = _word_overlap(query, memory_text)

    return {
        "cosine": round(base_cosine, 4),
        "salience": round(base_cosine * (1 + overlap * 0.3), 4),
        "semantic": round(base_cosine, 4),
        "recency": round(random.uniform(0.1, 0.5), 4),
        "goal_prox": round(random.uniform(0.0, 0.15), 4),
        "oq_boost": round(random.uniform(0.0, 0.1), 4),
        "narrative": round(random.uniform(0.0, 0.2), 4),
        "working_mode": round(random.uniform(0.0, 0.15), 4),
        "frequency": round(random.uniform(0.0, 0.2), 4),
        "entity_boost": round(random.uniform(0.0, 0.3) if is_relevant else 0.0, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Generator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_training_batch(memories, target_count=250):
    """Generate labeled training examples from real memories."""
    examples = []
    session_counter = 0

    # Shuffle memories for variety
    random.shuffle(memories)

    # Phase 1: Targeted queries matched to relevant memories (~60% useful)
    for query_text, keywords in TARGETED_QUERIES:
        # Find memories that match these keywords
        matching = []
        for mem in memories:
            text = mem.get("memory", "").lower()
            hits = sum(1 for kw in keywords if kw in text)
            if hits >= 1:
                matching.append((mem, hits))

        if not matching:
            continue

        # Sort by keyword match count, take top matches
        matching.sort(key=lambda x: x[1], reverse=True)

        session_counter += 1
        sid = f"synth-{session_counter:04d}"

        # Pick 2-5 results (simulating a real search returning multiple candidates)
        n_results = min(random.randint(2, 5), len(matching))
        selected = matching[:n_results]

        # Generate response using the top match
        best_mem = selected[0][0]
        response = generate_useful_response(best_mem.get("memory", ""), query_text)

        for rank, (mem, kw_hits) in enumerate(selected, 1):
            mem_text = mem.get("memory", "")
            is_relevant = kw_hits >= 2
            scores = sim_scores(mem_text, query_text, is_relevant)

            # Compute actual label from overlap with response
            overlap = _word_overlap(mem_text, response)
            if overlap > 0.5:
                label, reason = 1.0, f"useful: overlap={overlap:.2f}"
            elif overlap > 0.2:
                label, reason = 0.5, f"partial: overlap={overlap:.2f}"
            else:
                label, reason = 0.1, f"irrelevant: overlap={overlap:.2f}"

            examples.append({
                "type": "retrieval",
                "session_id": sid,
                "turn_number": 1,
                "query": query_text,
                "memory_id": mem.get("id", str(uuid4())),
                "memory_text": mem_text,
                "scores": scores,
                "gate_score": round(random.uniform(0.5, 1.0), 3),
                "working_mode": "general",
                "rank": rank,
                "label": label,
                "reason": reason,
            })

        examples.append({
            "type": "response",
            "session_id": sid,
            "turn_number": 1,
            "response_text": response,
        })

        if len([e for e in examples if e["type"] == "retrieval"]) >= target_count:
            break

    # Phase 2: Generic queries paired with random memories (~30% irrelevant)
    for gq in GENERIC_QUERIES:
        if len([e for e in examples if e["type"] == "retrieval"]) >= target_count:
            break

        session_counter += 1
        sid = f"synth-{session_counter:04d}"

        # Pick random memories (they won't be relevant to generic queries)
        n_results = random.randint(2, 4)
        selected = random.sample(memories, min(n_results, len(memories)))

        response = generate_irrelevant_response(gq)

        for rank, mem in enumerate(selected, 1):
            mem_text = mem.get("memory", "")
            scores = sim_scores(mem_text, gq, False)
            overlap = _word_overlap(mem_text, response)

            # Most should be irrelevant
            if overlap > 0.3:
                label, reason = 0.5, f"partial: overlap={overlap:.2f}"
            else:
                label, reason = 0.1, f"irrelevant: overlap={overlap:.2f}"

            examples.append({
                "type": "retrieval",
                "session_id": sid,
                "turn_number": 1,
                "query": gq,
                "memory_id": mem.get("id", str(uuid4())),
                "memory_text": mem_text,
                "scores": scores,
                "gate_score": round(random.uniform(0.3, 0.6), 3),
                "working_mode": "general",
                "rank": rank,
                "label": label,
                "reason": reason,
            })

        examples.append({
            "type": "response",
            "session_id": sid,
            "turn_number": 1,
            "response_text": response,
        })

    # Phase 3: Multi-turn sessions with partial responses (~10%)
    for i in range(5):
        if len([e for e in examples if e["type"] == "retrieval"]) >= target_count:
            break

        session_counter += 1
        sid = f"synth-{session_counter:04d}"

        # Pick a targeted query and follow up
        query_text, keywords = random.choice(TARGETED_QUERIES)

        matching = [m for m in memories if any(kw in m.get("memory", "").lower() for kw in keywords)]
        if not matching:
            continue

        selected = matching[:3]
        best_mem = selected[0]
        response = generate_partial_response(best_mem.get("memory", ""), query_text)

        for rank, mem in enumerate(selected, 1):
            mem_text = mem.get("memory", "")
            scores = sim_scores(mem_text, query_text, True)
            overlap = _word_overlap(mem_text, response)

            if overlap > 0.4:
                label, reason = 0.5, f"partial: overlap={overlap:.2f}"
            elif overlap > 0.15:
                label, reason = 0.5, f"partial: overlap={overlap:.2f}"
            else:
                label, reason = 0.1, f"irrelevant: overlap={overlap:.2f}"

            examples.append({
                "type": "retrieval",
                "session_id": sid,
                "turn_number": 1,
                "query": query_text,
                "memory_id": mem.get("id", str(uuid4())),
                "memory_text": mem_text,
                "scores": scores,
                "gate_score": round(random.uniform(0.4, 0.8), 3),
                "working_mode": "general",
                "rank": rank,
                "label": label,
                "reason": reason,
            })

        examples.append({
            "type": "response",
            "session_id": sid,
            "turn_number": 1,
            "response_text": response,
        })

    return examples


# ═══════════════════════════════════════════════════════════════════════════════
# Insert into PG
# ═══════════════════════════════════════════════════════════════════════════════

def insert_examples(examples, dry_run=False):
    """Insert generated examples into ml_retrievals and ml_responses."""
    retrievals = [e for e in examples if e["type"] == "retrieval"]
    responses = [e for e in examples if e["type"] == "response"]

    print(f"\n  Retrievals to insert: {len(retrievals)}")
    print(f"  Responses to insert:  {len(responses)}")

    # Count label distribution
    labels = {}
    for r in retrievals:
        l = r["label"]
        labels[l] = labels.get(l, 0) + 1
    print(f"  Label distribution:   {labels}")

    if dry_run:
        print("\n  [DRY RUN] No data inserted.")
        # Show a few examples
        for ex in retrievals[:3]:
            print(f"\n    Q: {ex['query'][:60]}")
            print(f"    M: {ex['memory_text'][:60]}")
            print(f"    L: {ex['label']} ({ex['reason']})")
            print(f"    S: cosine={ex['scores']['cosine']}, salience={ex['scores']['salience']}")
        return

    # Insert retrievals individually with parameterized queries
    inserted_r = 0
    for r in retrievals:
        s = r["scores"]
        qwc = len(r["query"].split())
        mwc = len(r["memory_text"].split())
        is_q = r["query"].rstrip().endswith("?")

        db.pg_execute(
            "INSERT INTO ml_retrievals ("
            "id, session_id, turn_number, query, memory_id, memory_text, "
            "cosine_score, salience_score, "
            "score_semantic, score_recency, score_goal_prox, score_oq_boost, "
            "score_narrative, score_working_mode, score_frequency, score_entity_boost, "
            "gate_score, working_mode, query_word_count, query_is_question, "
            "memory_word_count, result_rank, label, label_reason, labeled_at"
            ") VALUES ("
            "%s, %s, %s, %s, %s, %s, "
            "%s, %s, "
            "%s, %s, %s, %s, "
            "%s, %s, %s, %s, "
            "%s, %s, %s, %s, "
            "%s, %s, %s, %s, NOW())",
            (str(uuid4()), r['session_id'], r['turn_number'],
             r['query'], r['memory_id'], r['memory_text'],
             s['cosine'], s['salience'],
             s['semantic'], s['recency'], s['goal_prox'], s['oq_boost'],
             s['narrative'], s['working_mode'], s['frequency'], s['entity_boost'],
             r['gate_score'], r.get('working_mode', 'general'),
             qwc, is_q, mwc, r['rank'],
             r['label'], r['reason'])
        )
        inserted_r += 1

    # Insert responses
    inserted_resp = 0
    for r in responses:
        wc = len(r["response_text"].split())
        db.pg_execute(
            "INSERT INTO ml_responses (id, session_id, turn_number, response_text, response_word_count) "
            "VALUES (%s, %s, %s, %s, %s)",
            (str(uuid4()), r['session_id'], r['turn_number'],
             r['response_text'], wc)
        )
        inserted_resp += 1

    print(f"\n  Inserted {inserted_r} retrievals, {inserted_resp} responses")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    target = 250
    for arg in sys.argv:
        if arg.startswith("--count"):
            try:
                target = int(sys.argv[sys.argv.index(arg) + 1])
            except (ValueError, IndexError):
                pass

    print("  Fetching memories from Qdrant...")
    memories = fetch_all_memories(limit=300)
    print(f"  Got {len(memories)} memories")

    if not memories:
        print("  ERROR: No memories found. Is the daemon running?")
        sys.exit(1)

    print(f"  Generating ~{target} training examples...")
    examples = generate_training_batch(memories, target_count=target)

    insert_examples(examples, dry_run=dry_run)

    if not dry_run:
        # Show final stats
        from outcome_logger import get_stats
        stats = get_stats()
        print(f"\n  Total labeled: {stats['labeled']}/{stats['total_retrievals']}")
        print(f"  ML-1 ready: {'YES' if stats['salience_ready'] else 'NO'} ({stats['salience_progress']})")
