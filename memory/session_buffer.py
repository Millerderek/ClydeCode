#!/usr/bin/env python3
"""
session_buffer.py -- Redis-backed recent conversation buffer for OpenClaw.

Stores the last N exchanges (query + response snippets) per session.
When a new query comes in, the buffer is searched for relevant prior context
from the same session. Hits are proposed to the confidence gate as
"session_buffer" source type.

This handles the "what did we just talk about?" case — short-term memory
that doesn't need Mem0/Qdrant but is highly relevant within a session.

Usage:
    from session_buffer import SessionBuffer
    buf = SessionBuffer()
    buf.record(session_id, turn, query, response_snippet)
    hits = buf.search(session_id, query, limit=3)
"""

import json
import os
import re
import time

REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_SESSION_DB", "2"))  # Separate DB from cache

# How many turns to keep per session
MAX_TURNS_PER_SESSION = 20
# TTL for session data (4 hours — sessions don't last longer)
SESSION_TTL_SECONDS = 4 * 3600

# Stopwords for overlap matching
_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all",
    "can", "had", "her", "was", "one", "our", "out", "has",
    "his", "how", "its", "may", "new", "now", "old", "see",
    "way", "who", "did", "get", "let", "say", "she", "too",
    "use", "this", "that", "with", "have", "from", "they",
    "been", "some", "what", "when", "will", "more", "into",
    "also", "than", "them", "very", "just", "about", "which",
    "their", "there", "would", "could", "should", "where",
})

_redis = None


def _get_redis():
    """Lazy Redis connection."""
    global _redis
    if _redis is None:
        import redis
        _redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                             decode_responses=True, socket_timeout=2)
    return _redis


def _extract_words(text):
    """Extract meaningful words for matching."""
    return set(
        w for w in re.findall(r'[a-z]{3,}', text.lower())
        if w not in _STOPWORDS
    )


class SessionBuffer:
    """
    Redis-backed buffer for recent conversation turns.

    Storage format:
        Key: "sb:{session_id}" → JSON list of turn records
        Each record: {"turn": N, "query": "...", "response": "...", "ts": epoch}
    """

    def __init__(self):
        self._r = _get_redis()

    def _key(self, session_id):
        return f"sb:{session_id}"

    def record(self, session_id, turn_number, query, response_snippet=None):
        """
        Record a turn in the session buffer.

        Args:
            session_id: session identifier
            turn_number: turn within the session
            query: user query text
            response_snippet: first 500 chars of response (optional, added later)
        """
        if not session_id or not query:
            return

        key = self._key(session_id)

        # Load existing buffer
        raw = self._r.get(key)
        turns = json.loads(raw) if raw else []

        # Check if this turn already exists (update response if so)
        for t in turns:
            if t["turn"] == turn_number:
                if response_snippet:
                    t["response"] = response_snippet[:500]
                    t["ts"] = time.time()
                self._r.setex(key, SESSION_TTL_SECONDS, json.dumps(turns))
                return

        # New turn
        turns.append({
            "turn": turn_number,
            "query": query[:300],
            "response": (response_snippet or "")[:500],
            "ts": time.time(),
        })

        # Trim to max turns (keep most recent)
        if len(turns) > MAX_TURNS_PER_SESSION:
            turns = turns[-MAX_TURNS_PER_SESSION:]

        self._r.setex(key, SESSION_TTL_SECONDS, json.dumps(turns))

    def update_response(self, session_id, turn_number, response_snippet):
        """Update the response for an existing turn (without needing query)."""
        if not session_id or not response_snippet:
            return

        key = self._key(session_id)
        raw = self._r.get(key)
        if not raw:
            return

        turns = json.loads(raw)
        for t in turns:
            if t["turn"] == turn_number:
                t["response"] = response_snippet[:500]
                t["ts"] = time.time()
                self._r.setex(key, SESSION_TTL_SECONDS, json.dumps(turns))
                return

    def search(self, session_id, query, limit=3):
        """
        Search the session buffer for turns relevant to the current query.
        Uses word overlap scoring.

        Args:
            session_id: session identifier
            query: current query text
            limit: max results to return

        Returns:
            list of dicts: [{"text": "...", "score": float, "turn": int}]
        """
        if not session_id or not query:
            return []

        key = self._key(session_id)
        raw = self._r.get(key)
        if not raw:
            return []

        turns = json.loads(raw)
        if not turns:
            return []

        query_words = _extract_words(query)
        if not query_words:
            return []

        scored = []
        for t in turns:
            # Build combined text from query + response of that turn
            combined = (t.get("query", "") + " " + t.get("response", "")).strip()
            if not combined:
                continue

            turn_words = _extract_words(combined)
            if not turn_words:
                continue

            # Containment similarity: how much of the query matches this turn
            overlap = len(query_words & turn_words) / len(query_words)

            if overlap > 0.10:  # Minimum relevance
                # Build a readable summary
                text = f"[Turn {t['turn']}] Q: {t.get('query', '')[:100]}"
                resp = t.get("response", "")
                if resp:
                    text += f" → {resp[:200]}"

                scored.append({
                    "text": text,
                    "score": round(overlap, 4),
                    "turn": t["turn"],
                })

        # Sort by score descending, return top N
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def clear(self, session_id):
        """Clear a session's buffer."""
        self._r.delete(self._key(session_id))

    def stats(self, session_id=None):
        """Get buffer stats."""
        if session_id:
            raw = self._r.get(self._key(session_id))
            turns = json.loads(raw) if raw else []
            return {"session_id": session_id, "turns": len(turns)}

        # Count all session buffers
        keys = self._r.keys("sb:*")
        return {"active_sessions": len(keys)}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    buf = SessionBuffer()

    if len(sys.argv) < 2:
        print("Usage: session_buffer.py stats|test")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "stats":
        print(json.dumps(buf.stats(), indent=2))

    elif cmd == "test":
        sid = "test-session-001"
        buf.clear(sid)

        # Simulate a conversation
        buf.record(sid, 1, "How do I set up Docker Compose?",
                   "Use a docker-compose.yml file with services defined...")
        buf.record(sid, 2, "What about persistent volumes?",
                   "Add volumes section to your service, mount to host path...")
        buf.record(sid, 3, "How do I restart containers?",
                   "docker compose restart, or docker compose up -d to recreate...")

        # Search
        hits = buf.search(sid, "Docker volume configuration", limit=3)
        print(f"\nSearch: 'Docker volume configuration' in session {sid}")
        for h in hits:
            print(f"  [{h['score']:.3f}] {h['text'][:80]}...")

        hits2 = buf.search(sid, "restart the containers", limit=3)
        print(f"\nSearch: 'restart the containers' in session {sid}")
        for h in hits2:
            print(f"  [{h['score']:.3f}] {h['text'][:80]}...")

        buf.clear(sid)
        print("\nTest complete.")
