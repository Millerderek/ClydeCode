#!/usr/bin/env python3
"""
session_ingest — Auto-ingest session summaries into ClydeMemory

Watches the session memory directory for session markdown files.
Tracks which files have been ingested in a state file.
Extracts key facts from conversations and stores them as memories.

# Requires clyde-memory full stack (memo_daemon + topic_classifier)

Usage:
  session_ingest.py              # Process new session files
  session_ingest.py --force      # Re-process all files
  session_ingest.py --status     # Show ingestion status
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from daemon_client import memo_add
from topic_classifier import classify_and_store_fact

from config import SESSION_MEMORY_DIR as MEMORY_DIR
from config import CLYDE_STATE_DIR
from config import CLYDE_USER as USER

STATE_FILE = CLYDE_STATE_DIR / ".session_ingest_state.json"

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"ingested": {}}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def extract_facts(content):
    """Extract meaningful facts from a session transcript."""
    facts = []
    lines = content.split("\n")

    # Get assistant responses (they contain the useful info)
    current_block = []
    in_assistant = False

    for line in lines:
        if line.startswith("assistant:"):
            in_assistant = True
            current_block = [line[len("assistant:"):].strip()]
        elif line.startswith("user:"):
            if in_assistant and current_block:
                text = " ".join(current_block)
                # Clean up formatting
                text = re.sub(r'\[\[.*?\]\]', '', text)
                text = re.sub(r'\*\*', '', text)
                text = re.sub(r'`[^`]+`', '', text)
                text = text.strip()
                if len(text) > 50:
                    facts.append(text)
            in_assistant = False
            current_block = []
        elif in_assistant:
            current_block.append(line.strip())

    # Capture last block
    if in_assistant and current_block:
        text = " ".join(current_block)
        text = re.sub(r'\[\[.*?\]\]', '', text)
        text = re.sub(r'\*\*', '', text)
        text = text.strip()
        if len(text) > 50:
            facts.append(text)

    return facts

def summarize_fact(fact):
    """Condense a long assistant response into a storable memory."""
    # If it's short enough, use as-is
    if len(fact) < 200:
        return fact

    # Extract bullet points as individual facts
    bullets = re.findall(r'[-•]\s+(.+?)(?=\n[-•]|\n\n|$)', fact)
    if bullets:
        return bullets

    # Truncate long text to first meaningful sentence
    sentences = re.split(r'[.!?]\s+', fact)
    if sentences:
        return sentences[0][:200]

    return fact[:200]

def ingest_file(filepath, state):
    """Ingest a session file into ClydeMemory."""
    content = filepath.read_text()
    file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    # Skip if already ingested with same content
    fname = filepath.name
    if fname in state["ingested"] and state["ingested"][fname] == file_hash:
        return 0

    facts = extract_facts(content)
    stored = 0

    for fact in facts:
        summaries = summarize_fact(fact)
        if isinstance(summaries, list):
            for s in summaries:
                if len(s.strip()) > 20:
                    store_memory(s.strip())
                    stored += 1
        elif isinstance(summaries, str) and len(summaries.strip()) > 20:
            store_memory(summaries.strip())
            stored += 1

    state["ingested"][fname] = file_hash
    return stored

def store_memory(text):
    """Store a single fact via daemon socket (falls back to CLI)."""
    if not memo_add(text, user_id=USER, writer="session-ingest"):
        print(f"  WARNING: Failed to store: {text[:60]}")
        return
    # Also classify and store in topic system
    try:
        tr = classify_and_store_fact(text, source="session_ingest")
        if tr.get("stored"):
            print(f"    -> topic: {tr['topic_slug']} ({tr['method']})")
    except Exception:
        pass  # Topic storage is best-effort

def cmd_ingest(force=False):
    """Process new session files."""
    state = load_state() if not force else {"ingested": {}}

    if not MEMORY_DIR.exists():
        print(f"  WARNING: Memory directory not found: {MEMORY_DIR}")
        return

    files = sorted(MEMORY_DIR.glob("*.md"))
    if not files:
        print("  No session files found")
        return

    total = 0
    for f in files:
        count = ingest_file(f, state)
        if count > 0:
            print(f"  OK {f.name}: {count} facts ingested")
            total += count
        else:
            print(f"  . {f.name}: already ingested")

    save_state(state)
    print(f"\n  Total: {total} new facts ingested from {len(files)} files")

def cmd_status():
    """Show ingestion status."""
    state = load_state()
    files = sorted(MEMORY_DIR.glob("*.md")) if MEMORY_DIR.exists() else []

    print(f"\n  Session Ingestion Status")
    print(f"  {'=' * 40}")
    print(f"  Memory dir: {MEMORY_DIR}")
    print(f"  Files found: {len(files)}")
    print(f"  Files ingested: {len(state.get('ingested', {}))}")

    for f in files:
        status = "OK" if f.name in state.get("ingested", {}) else "."
        print(f"  {status} {f.name}")

if __name__ == "__main__":
    if "--status" in sys.argv:
        cmd_status()
    elif "--force" in sys.argv:
        cmd_ingest(force=True)
    else:
        cmd_ingest()
