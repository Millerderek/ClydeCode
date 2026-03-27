#!/usr/bin/env python3
"""
memo_daemon.py -- Persistent Unix socket daemon for ClydeMemory operations.

Holds a warm Mem0 Memory instance so callers skip the ~7s cold start.
Listens on a Unix socket, accepts JSON-line requests.

Usage:
    memo_daemon.py start       # Daemonize and run
    memo_daemon.py stop        # Stop via PID file
    memo_daemon.py status      # Check if running
    memo_daemon.py foreground  # Run in foreground (debug)
"""

import json
import os
import signal
import socket
import socketserver
import sys
import time
import threading
import traceback
from pathlib import Path

# Kill telemetry before mem0 import
os.environ.setdefault("MEM0_TELEMETRY", "false")

from config import CLYDE_USER, CLYDE_LOG_DIR

SOCK_PATH = os.environ.get("CLYDE_SOCK_PATH", "/tmp/clyde-memo.sock")
PID_FILE = "/tmp/clyde-memo-daemon.pid"
LOG_FILE = str(CLYDE_LOG_DIR / "clyde-memo-daemon.log")
MAX_REQUEST_SIZE = 1 * 1024 * 1024  # 1MB

# Add our directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ===============================================================================
# Globals (set during init)
# ===============================================================================

_memory = None
_start_time = None
_request_count = 0
_confidence_gate = None
_session_buffer = None  # Lazy-initialized SessionBuffer instance

# CG-1: Track admitted proposals per turn for earn-back evaluation.
# Key: (session_id, turn_number) → list of (source_type, content_snippet)
# Ring buffer — keep last 200 turns max.
_injection_buffer = {}
_INJECTION_BUFFER_MAX = 200
_injection_lock = threading.Lock()  # Guards _injection_buffer + _confidence_gate


def _log(msg):
    """Log with timestamp to stderr (captured by daemonize)."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def init_memory():
    """Initialize Mem0 Memory instance (one-time cost)."""
    global _memory, _start_time
    _log("Initializing Memory...")
    t0 = time.time()

    from openclaw_memo import get_memo_config
    from mem0 import Memory
    config = get_memo_config()
    _memory = Memory.from_config(config)
    _start_time = time.time()

    elapsed = time.time() - t0
    _log(f"Memory initialized in {elapsed:.2f}s")
    return _memory


# ===============================================================================
# Helpers
# ===============================================================================

def _derive_topic_tags(query: str, entities: list[str] | None = None) -> list[str]:
    """Derive topic tags from query text and extracted entities.

    Combines entity names with significant query keywords for use in
    topic-aware pattern matching. Returns lowercase, deduplicated tags.
    """
    tags = set()

    # Add entity names (already extracted by entity_boost)
    if entities:
        for e in entities:
            tags.add(e.lower().strip())

    # Extract significant keywords from query (skip stopwords)
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "just", "about", "up", "it", "its",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
        "she", "her", "they", "them", "their", "and", "but", "or", "if",
    }
    import re
    words = re.findall(r"[a-zA-Z0-9_-]+", query.lower())
    for w in words:
        if len(w) > 2 and w not in stopwords:
            tags.add(w)

    return list(tags)[:20]  # Cap at 20 tags


# ===============================================================================
# Request handlers
# ===============================================================================

def handle_search(params):
    """Search with entity-boosted scoring + context gate."""
    query = params.get("query", "")
    user_id = params.get("user_id", CLYDE_USER)
    limit = params.get("limit", 5)
    skip_gate = params.get("skip_gate", False)

    if not query:
        return {"ok": False, "error": "Missing 'query' parameter"}

    # Context Need Gate: skip retrieval for self-contained prompts
    gate_score = 1.0
    if not skip_gate:
        try:
            from context_gate import needs_context, THRESHOLD
            gate_score = needs_context(query)
            if gate_score < THRESHOLD:
                return {"ok": True, "results": [], "gated": True,
                        "gate_score": gate_score, "elapsed_ms": 0}
        except Exception:
            pass  # Gate failed, proceed with search

    # Fetch wider semantic results for hybrid merge
    results = _memory.search(query, user_id=user_id, limit=max(limit * 2, 10))
    semantic_results = results.get("results", [])

    # ── Graph-enhanced retrieval: inject neighbor memories ──
    # Two paths: (1) graph entities+relationships for hop traversal
    #            (2) memory_entities co-occurrence for direct entity sharing
    try:
        from entity_boost import extract_entities_from_text
        from graph_extractor import _run_pg as graph_pg

        query_entities = list(extract_entities_from_text(query))
        if query_entities:
            semantic_ids = {r.get("id", "") for r in semantic_results}
            graph_injected = 0

            ent_names = query_entities[:5]

            # Path 1: memory_entities co-occurrence — find memories that share
            # entities with the query but weren't in semantic results.
            # Ranked by how many query entities they share.
            # Uses ANY() with ILIKE via unnest for safe parameterization.
            cooccur_sql = """
                SELECT m.qdrant_point_id, m.summary, COUNT(DISTINCT me.entity_name) as shared
                FROM memory_entities me
                JOIN memories m ON me.memory_id = m.id
                WHERE me.entity_name ILIKE ANY(%s)
                  AND m.is_deprecated = FALSE
                GROUP BY m.qdrant_point_id, m.summary
                ORDER BY shared DESC
                LIMIT 10;
            """
            cooccur_rows = graph_pg(cooccur_sql, (ent_names,))

            # Path 2: graph hop traversal — find memories linked to entities
            # connected to query entities via relationships.
            hop_sql = """
                SELECT DISTINCT m.qdrant_point_id, m.summary
                FROM entities e
                JOIN relationships r ON r.source_id = e.id OR r.target_id = e.id
                JOIN entities e2 ON e2.id = CASE
                    WHEN r.source_id = e.id THEN r.target_id
                    ELSE r.source_id END
                JOIN memory_entity_links mel ON mel.entity_id = e2.id
                JOIN memories m ON m.id = mel.memory_id
                WHERE e.name ILIKE ANY(%s)
                  AND m.is_deprecated = FALSE
                LIMIT 10;
            """
            hop_rows = graph_pg(hop_sql, (ent_names,))

            # Merge both paths
            for rows in (cooccur_rows, hop_rows):
                if not rows:
                    continue
                for line in rows.split("\n"):
                    if "|" not in line:
                        continue
                    parts = line.split("|")
                    qid = parts[0].strip()
                    summary = parts[1].strip() if len(parts) > 1 else ""
                    if qid and qid not in semantic_ids and summary and graph_injected < 5:
                        semantic_results.append({
                            "memory": summary,
                            "score": 0.25,
                            "id": qid,
                            "_source": "graph",
                        })
                        semantic_ids.add(qid)
                        graph_injected += 1

            if graph_injected:
                _log(f"[graph-retrieval] Injected {graph_injected} neighbor memories for: {query[:60]}")
    except Exception as e:
        _log(f"[graph-retrieval] Skipped: {e}")

    # Salience scoring: Phase 2+ upgraded engine (falls back to topic_scorer)
    rerank_strategy = params.get("rerank_strategy", "mmr")  # default: MMR diversity pass
    try:
        from salience_engine import salience_score
        boosted = salience_score(query, semantic_results, limit=limit)
        # Apply post-scoring reranker (MMR by default for diversity)
        if rerank_strategy and boosted and len(boosted) > 1:
            try:
                from zep_features import rerank
                boosted = rerank(boosted, strategy=rerank_strategy,
                                limit=limit, query=query)
            except Exception as e:
                _log(f"[rerank] {rerank_strategy} failed, keeping salience order: {e}")
    except Exception:
        try:
            from topic_scorer import hybrid_search
            boosted = hybrid_search(semantic_results, query, limit=limit)
        except Exception:
            # Final fallback: simple semantic + entity boost
            try:
                from entity_boost import get_boost
            except ImportError:
                get_boost = lambda q, mid: 0.0

            boosted = []
            for r in semantic_results[:limit]:
                memory = r.get("memory", "")
                score = r.get("score", 0)
                mem_id = r.get("id", "")
                short_id = mem_id[:8]
                boost = get_boost(query, short_id)
                boosted.append({
                    "final": round(score + boost, 4),
                    "score": round(score, 4),
                    "boost": round(boost, 4),
                    "lexical": 0.0,
                    "memory": memory,
                    "mem_id": short_id,
                    "id": mem_id,
                })
            boosted.sort(key=lambda x: x["final"], reverse=True)

    if not boosted:
        return {"ok": True, "results": [], "gate_score": gate_score, "elapsed_ms": 0}

    # Track access
    result_ids = [r["id"] for r in boosted if r.get("id")]
    if result_ids:
        try:
            from openclaw_memo import _track_access
            _track_access(result_ids)
        except Exception:
            pass

    # ── ML-0: Outcome logging (fire-and-forget) ──
    try:
        from outcome_logger import log_retrieval
        threading.Thread(
            target=log_retrieval,
            kwargs={
                "query": query,
                "results": boosted,
                "gate_score": gate_score,
                "session_id": params.get("session_id"),
                "turn_number": params.get("turn_number", 0),
            },
            daemon=True,
        ).start()
    except Exception:
        pass

    # ── Cognitive Architecture context assembly ──
    # Each subsystem proposes context. The confidence gate decides what gets in.

    try:
        from confidence_gate import (
            ConfidenceGate, ContextProposal,
            classify_query_complexity, estimate_confidence_for_ca,
            score_memo_results,
        )

        # Get or create the session gate (persists across turns in daemon)
        global _confidence_gate
        if "_confidence_gate" not in globals() or _confidence_gate is None:
            _confidence_gate = ConfidenceGate()
        cg = _confidence_gate

        complexity = classify_query_complexity(query)
        proposals = []

        # Memo search results as proposals
        proposals.extend(score_memo_results(boosted))

        # Phase 1: Graph context
        try:
            from entity_boost import extract_entities_from_text
            from graph_extractor import build_graph_context
            query_entities = list(extract_entities_from_text(query))
            if query_entities:
                gc = build_graph_context(query_entities[:5])
                if gc:
                    proposals.append(ContextProposal(
                        "graph", gc,
                        confidence=estimate_confidence_for_ca("graph", gc, query),
                    ))
        except Exception:
            query_entities = []

        # Phase 7A: Cluster context (graph evolution)
        try:
            from graph_evolution import get_cluster_context
            if query_entities:
                cc = get_cluster_context(query_entities[:3])
                if cc:
                    proposals.append(ContextProposal(
                        "cluster", cc,
                        confidence=estimate_confidence_for_ca("cluster", cc, query),
                    ))
        except Exception:
            pass

        # Phase 3: Narrative context
        try:
            from narrative_engine import get_narrative_context
            nc = get_narrative_context(
                entity_names=query_entities[:5] if query_entities else None
            )
            if nc:
                proposals.append(ContextProposal(
                    "narrative", nc,
                    confidence=estimate_confidence_for_ca("narrative", nc, query),
                ))
        except Exception:
            pass

        # Phase 4: Peripheral awareness (pending signals)
        try:
            from peripheral_awareness import build_awareness_context
            ac = build_awareness_context(limit=2)
            if ac:
                proposals.append(ContextProposal(
                    "peripheral", ac,
                    confidence=estimate_confidence_for_ca("peripheral", ac, query),
                ))
        except Exception:
            pass

        # Phase 5: Learned patterns (topic-aware)
        try:
            from abstraction_engine import build_topic_patterns_context
            topic_tags = _derive_topic_tags(query, query_entities)
            pc = build_topic_patterns_context(query, topic_tags=topic_tags, limit=2)
            if pc:
                proposals.append(ContextProposal(
                    "patterns", pc,
                    confidence=estimate_confidence_for_ca("patterns", pc, query),
                ))
        except Exception:
            pass

        # Phase 7B: Episodic context (past work sessions + causal chains)
        try:
            from episodic_memory import build_episode_context
            ec = build_episode_context(
                query,
                entity_names=query_entities[:5] if query_entities else None,
                limit=2,
            )
            if ec:
                proposals.append(ContextProposal(
                    "episodic", ec,
                    confidence=estimate_confidence_for_ca("episodic", ec, query),
                ))
        except Exception:
            pass

        # Phase 6: Identity snapshot (lightweight — only era + season layers)
        try:
            from soul_growth import build_identity_context
            ic = build_identity_context()
            if ic:
                proposals.append(ContextProposal(
                    "identity", ic,
                    confidence=estimate_confidence_for_ca("identity", ic, query),
                ))
        except Exception:
            pass

        # Phase 7C: Predictive context (mental models + anticipation)
        try:
            from mental_models import build_predictive_context
            pc = build_predictive_context(
                query,
                entity_names=query_entities[:5] if query_entities else None,
            )
            if pc:
                proposals.append(ContextProposal(
                    "predictive", pc,
                    confidence=estimate_confidence_for_ca("predictive", pc, query),
                ))
        except Exception:
            pass

        # Session buffer: recent turns from the same session
        # Search FIRST (before recording current query to avoid self-echo)
        session_id = params.get("session_id")
        turn_number_sb = params.get("turn_number", 0)
        if session_id:
            try:
                global _session_buffer
                if _session_buffer is None:
                    from session_buffer import SessionBuffer
                    _session_buffer = SessionBuffer()
                hits = _session_buffer.search(session_id, query, limit=3)
                for hit in hits:
                    # Skip hits from the current turn (self-echo)
                    if hit.get("turn") == turn_number_sb:
                        continue
                    proposals.append(ContextProposal(
                        "session_buffer",
                        hit["text"],
                        confidence=hit["score"],
                    ))
                # NOW record this query for future turns
                _session_buffer.record(session_id, turn_number_sb, query)
            except Exception:
                pass

        # ── GATE ──
        # Build session context for adaptive thresholds
        session_context = {
            "turn_number": params.get("turn_number", 1),
        }
        # Deadline pressure (lowers goal-related thresholds when deadline is near)
        try:
            from goal_tracker import get_nearest_deadline_days
            dd = get_nearest_deadline_days()
            if dd is not None:
                session_context["nearest_deadline_days"] = dd
        except Exception:
            pass
        # Correction rate (raises all thresholds when corrections are frequent)
        try:
            from outcome_logger import get_correction_rate
            session_context["correction_rate_7d"] = get_correction_rate(days=7)
        except Exception:
            pass
        admitted = cg.gate(proposals, query_complexity=complexity,
                           session_context=session_context)

        # Build ca_context from admitted proposals (excluding memo_search,
        # which is already in 'results')
        ca_context = {}
        source_key_map = {
            "graph": "graph", "narrative": "narratives",
            "peripheral": "awareness", "patterns": "patterns",
            "identity": "identity",
        }
        for p in admitted:
            key = source_key_map.get(p.source_type)
            if key:
                ca_context[key] = p.content

        # CG-1: Record what was injected for earn-back evaluation
        _record_injection(
            session_id=params.get("session_id"),
            turn_number=params.get("turn_number", 0),
            admitted=admitted,
        )

        # Filter memo results to only those that passed the gate
        admitted_memo_ids = set()
        for p in admitted:
            if p.source_type == "memo_search":
                mid = p.metadata.get("id", "")
                if mid:
                    admitted_memo_ids.add(mid)

        if admitted_memo_ids:
            boosted = [r for r in boosted if r.get("id") in admitted_memo_ids]

        resp = {"ok": True, "results": boosted, "gate_score": gate_score,
                "query_complexity": complexity,
                "gate_stats": {
                    "proposed": len(proposals),
                    "admitted": len(admitted),
                    "tokens_used": sum(p.token_estimate for p in admitted),
                }}
        if ca_context:
            resp["ca_context"] = ca_context
        return resp

    except ImportError:
        _log("[confidence_gate] Module not available, falling back to ungated assembly")
    except Exception as e:
        _log(f"[confidence_gate] Gate error: {e}, falling back to ungated assembly")

    # ── Fallback: ungated assembly (same as before) ──
    ca_context = {}
    try:
        from entity_boost import extract_entities_from_text
        from graph_extractor import build_graph_context
        query_entities = list(extract_entities_from_text(query))
        if query_entities:
            gc = build_graph_context(query_entities[:5])
            if gc:
                ca_context["graph"] = gc
    except Exception:
        query_entities = []

    try:
        from narrative_engine import get_narrative_context
        nc = get_narrative_context(
            entity_names=query_entities[:5] if query_entities else None
        )
        if nc:
            ca_context["narratives"] = nc
    except Exception:
        pass

    try:
        from peripheral_awareness import build_awareness_context
        ac = build_awareness_context(limit=2)
        if ac:
            ca_context["awareness"] = ac
    except Exception:
        pass

    try:
        from abstraction_engine import build_patterns_context
        pc = build_patterns_context(query, limit=2)
        if pc:
            ca_context["patterns"] = pc
    except Exception:
        pass

    try:
        from soul_growth import build_identity_context
        ic = build_identity_context()
        if ic:
            ca_context["identity"] = ic
    except Exception:
        pass

    resp = {"ok": True, "results": boosted, "gate_score": gate_score}
    if ca_context:
        resp["ca_context"] = ca_context
    return resp


def handle_add(params):
    """Add a memory with optional impact scoring."""
    text = params.get("text", "")
    user_id = params.get("user_id", CLYDE_USER)
    impact = params.get("impact", "auto")

    if not text:
        return {"ok": False, "error": "Missing 'text' parameter"}

    # Auto-detect impact
    if impact == "auto":
        try:
            from openclaw_memo import _detect_impact
            impact = _detect_impact(text)
        except Exception:
            impact = "normal"

    result = _memory.add(text, user_id=user_id)

    # Set impact in PG
    if impact != "normal" and result and result.get("results"):
        try:
            from openclaw_memo import _set_impact_pg
            for r in result["results"]:
                mem_id = r.get("id", "")
                if mem_id:
                    _set_impact_pg(mem_id, impact)
        except Exception:
            pass

    # Async graph extraction (fire-and-forget)
    if result and result.get("results"):
        for r in result["results"]:
            mem_id = r.get("id", "")
            if mem_id and text:
                threading.Thread(
                    target=_async_graph_extract,
                    args=(text, mem_id),
                    daemon=True,
                ).start()

    return {"ok": True, "result": result}


def _async_graph_extract(text: str, memory_id: str):
    """Fire-and-forget graph entity extraction after memory add."""
    try:
        from graph_extractor import extract_and_store
        result = extract_and_store(text, memory_id)
        if result.get("triples", 0) > 0:
            _log(f"[graph] Extracted {result['triples']} triples from {memory_id[:8]}")
    except Exception as e:
        _log(f"[graph] Extraction failed for {memory_id[:8]}: {e}")


# ===============================================================================
# CG-1: Earn-back — track injections and evaluate against responses
# ===============================================================================

def _record_injection(session_id, turn_number, admitted):
    """Store what was injected this turn for later earn-back evaluation."""
    global _injection_buffer
    if not session_id or not admitted:
        return
    key = (session_id, turn_number)
    with _injection_lock:
        # Store source_type + first 500 chars of content (enough for overlap check)
        _injection_buffer[key] = [
            (p.source_type, p.content[:500]) for p in admitted
        ]
        # Evict oldest entries if buffer is full
        if len(_injection_buffer) > _INJECTION_BUFFER_MAX:
            oldest_keys = sorted(_injection_buffer.keys(), key=lambda k: k[1])
            for k in oldest_keys[:len(_injection_buffer) - _INJECTION_BUFFER_MAX]:
                del _injection_buffer[k]


def _evaluate_earnback(session_id, turn_number, response_text):
    """
    Check which injected sources were actually used in the response.
    Feeds results back to the confidence gate's earn-back mechanism.
    Called from daemon threads — uses _injection_lock for thread safety.
    """
    global _confidence_gate, _injection_buffer
    if not _confidence_gate or not session_id or not response_text:
        return

    key = (session_id, turn_number)
    with _injection_lock:
        injected = _injection_buffer.pop(key, None)
    if not injected:
        return

    # Word overlap evaluation (same approach as outcome_logger)
    import re as _re
    _stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all",
        "can", "had", "her", "was", "one", "our", "out", "has",
        "his", "how", "its", "may", "new", "now", "old", "see",
        "way", "who", "did", "get", "let", "say", "she", "too",
        "use", "this", "that", "with", "have", "from", "they",
        "been", "some", "what", "when", "will", "more", "into",
        "also", "than", "them", "very", "just", "about", "which",
    }

    resp_words = set(
        w for w in _re.findall(r'[a-z]{3,}', response_text.lower())
        if w not in _stopwords
    )
    if not resp_words:
        return

    earned = 0
    missed = 0
    for source_type, content in injected:
        content_words = set(
            w for w in _re.findall(r'[a-z]{3,}', content.lower())
            if w not in _stopwords
        )
        if not content_words:
            continue

        overlap = len(content_words & resp_words) / len(content_words)
        was_useful = overlap > 0.20  # 20% word overlap = "used"

        with _injection_lock:
            _confidence_gate.record_outcome(source_type, was_useful)
        if was_useful:
            earned += 1
        else:
            missed += 1

    if earned or missed:
        _log(f"[earn-back] session={session_id[:8] if session_id else '?'} "
             f"turn={turn_number}: {earned} useful, {missed} ignored")


def handle_get_all(params):
    """Get all memories for a user."""
    user_id = params.get("user_id", CLYDE_USER)
    results = _memory.get_all(user_id=user_id)
    return {"ok": True, "results": results.get("results", [])}


def handle_delete(params):
    """Delete a specific memory."""
    memory_id = params.get("memory_id", "")
    if not memory_id:
        return {"ok": False, "error": "Missing 'memory_id' parameter"}
    _memory.delete(memory_id)
    return {"ok": True}


def handle_delete_all(params):
    """Delete all memories for a user."""
    user_id = params.get("user_id", CLYDE_USER)
    _memory.delete_all(user_id=user_id)
    return {"ok": True}


def handle_status(params):
    """Return daemon health info."""
    uptime = time.time() - _start_time if _start_time else 0
    try:
        from topic_store import get_stats
        topic_stats = get_stats()
    except Exception:
        topic_stats = {}
    return {
        "ok": True,
        "uptime_s": round(uptime, 1),
        "requests_served": _request_count,
        "pid": os.getpid(),
        "topic_stats": topic_stats,
    }


# ===============================================================================
# Topic handlers (structured records)
# ===============================================================================

def handle_search_topics(params):
    """Search topics by FTS or entity overlap."""
    from topic_store import search_topics, search_topics_by_entities
    query = params.get("query", "")
    entities = params.get("entities", [])
    limit = params.get("limit", 5)

    if not query and not entities:
        return {"ok": False, "error": "Need 'query' or 'entities' parameter"}

    if entities:
        results = search_topics_by_entities(entities, limit=limit)
    else:
        results = search_topics(query, limit=limit)

    return {"ok": True, "results": results}


def handle_get_topic_context(params):
    """Get full topic context for prompt injection."""
    from topic_store import get_topic_context, track_topic_access
    slug = params.get("slug", "")
    topic_id = params.get("topic_id")

    if not slug and not topic_id:
        return {"ok": False, "error": "Need 'slug' or 'topic_id' parameter"}

    ctx = get_topic_context(slug=slug, topic_id=topic_id)
    if not ctx:
        return {"ok": False, "error": "Topic not found"}

    # Track access
    try:
        track_topic_access(ctx["topic"]["id"])
    except Exception:
        pass

    return {"ok": True, **ctx}


def handle_upsert_topic(params):
    """Create or update a topic snapshot."""
    from topic_store import upsert_topic
    title = params.get("title", "")
    summary = params.get("summary", "")
    entities = params.get("entities", [])
    scope = params.get("scope", "project")
    slug = params.get("slug")

    if not title:
        return {"ok": False, "error": "Missing 'title' parameter"}

    topic_id = upsert_topic(title=title, summary=summary, entities=entities,
                            scope=scope, slug=slug)
    return {"ok": True, "topic_id": topic_id}


def handle_add_fact(params):
    """Add a fact to a topic."""
    from topic_store import add_fact
    topic_id = params.get("topic_id")
    content = params.get("content", "")
    source = params.get("source", "auto")
    memory_id = params.get("memory_id")

    if not topic_id or not content:
        return {"ok": False, "error": "Need 'topic_id' and 'content' parameters"}

    fact_id = add_fact(topic_id=topic_id, content=content,
                       source=source, memory_id=memory_id)
    if fact_id == -1:
        return {"ok": True, "duplicate": True, "fact_id": -1}
    return {"ok": True, "fact_id": fact_id}


def handle_list_topics(params):
    """List all topics."""
    from topic_store import list_topics
    limit = params.get("limit", 50)
    return {"ok": True, "topics": list_topics(limit=limit)}


def handle_list_open_loops(params):
    """Get open loops, optionally filtered by topic."""
    from topic_store import get_open_loops
    topic_id = params.get("topic_id")
    status = params.get("status", "open")
    results = get_open_loops(topic_id=topic_id, status=status)
    return {"ok": True, "loops": results}


def handle_topic_stats(params):
    """Get topic system stats."""
    from topic_store import get_stats
    return {"ok": True, **get_stats()}


def handle_log_outcome(params):
    """ML-0: Log assistant response + CG-1 earn-back evaluation."""
    response_text = params.get("response_text", "")
    session_id = params.get("session_id")
    turn_number = params.get("turn_number", 0)

    if not response_text:
        return {"ok": False, "error": "Missing 'response_text' parameter"}

    # ML-0: Store response for retroactive labeling
    try:
        from outcome_logger import log_response
        threading.Thread(
            target=log_response,
            kwargs={
                "response_text": response_text,
                "session_id": session_id,
                "turn_number": turn_number,
            },
            daemon=True,
        ).start()
    except Exception:
        pass

    # CG-1: Evaluate earn-back (which injected sources were used?)
    try:
        threading.Thread(
            target=_evaluate_earnback,
            args=(session_id, turn_number, response_text),
            daemon=True,
        ).start()
    except Exception:
        pass

    # Session buffer: record response snippet for future turn lookups
    if session_id:
        try:
            global _session_buffer
            if _session_buffer is None:
                from session_buffer import SessionBuffer
                _session_buffer = SessionBuffer()
            _session_buffer.update_response(
                session_id, turn_number, response_text[:500]
            )
        except Exception:
            pass

    return {"ok": True}


def handle_ml_stats(params):
    """ML-0: Return outcome logger statistics."""
    try:
        from outcome_logger import get_stats
        return {"ok": True, "stats": get_stats()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def handle_ml_label(params):
    """ML-0: Run a labeling pass on collected retrievals."""
    batch_size = params.get("batch_size", 100)
    try:
        from outcome_logger import compute_labels
        result = compute_labels(batch_size=batch_size)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def handle_gate_stats(params):
    """CG-0: Return confidence gate statistics."""
    global _confidence_gate
    if "_confidence_gate" not in globals() or _confidence_gate is None:
        return {"ok": True, "stats": {"turns": 0, "message": "Gate not yet initialized"}}
    return {"ok": True, "stats": _confidence_gate.stats()}


def handle_gate_record_outcome(params):
    """CG-0: Record earn-back outcome for a source type."""
    global _confidence_gate
    if "_confidence_gate" not in globals() or _confidence_gate is None:
        return {"ok": False, "error": "Gate not yet initialized"}
    source_type = params.get("source_type", "")
    was_useful = params.get("was_useful", False)
    if not source_type:
        return {"ok": False, "error": "Missing 'source_type' parameter"}
    _confidence_gate.record_outcome(source_type, was_useful)
    return {"ok": True}


# ── LCM handler shims (lazy import to avoid startup cost) ────────────────────

_lcm_module = None

def _lcm_lazy():
    global _lcm_module
    if _lcm_module is None:
        try:
            import lcm_engine
            _lcm_module = lcm_engine
            _log("LCM engine loaded")
        except Exception as e:
            _log(f"LCM engine load failed: {e}")
    return _lcm_module

def _lcm_search(params):
    lcm = _lcm_lazy()
    if lcm is None:
        return {"ok": False, "error": "LCM engine not available"}
    return lcm.handle_lcm_search(params)

def _lcm_context(params):
    lcm = _lcm_lazy()
    if lcm is None:
        return {"ok": False, "error": "LCM engine not available"}
    return lcm.handle_lcm_context(params)

def _lcm_stats(params):
    lcm = _lcm_lazy()
    if lcm is None:
        return {"ok": False, "error": "LCM engine not available"}
    return lcm.handle_lcm_stats(params)

def _lcm_pressure(params):
    lcm = _lcm_lazy()
    if lcm is None:
        return {"ok": False, "error": "LCM engine not available"}
    return lcm.handle_lcm_pressure(params)

def _lcm_grep(params):
    lcm = _lcm_lazy()
    if lcm is None:
        return {"ok": False, "error": "LCM engine not available"}
    return lcm.handle_lcm_grep(params)

def _lcm_describe(params):
    lcm = _lcm_lazy()
    if lcm is None:
        return {"ok": False, "error": "LCM engine not available"}
    return lcm.handle_lcm_describe(params)

def _lcm_recall(params):
    lcm = _lcm_lazy()
    if lcm is None:
        return {"ok": False, "error": "LCM engine not available"}
    return lcm.handle_lcm_recall(params)


# ── Graph Evolution handler shims (Phase 7A) ──────────────────────────────────

_graph_evo_module = None

def _graph_evo_lazy():
    global _graph_evo_module
    if _graph_evo_module is None:
        try:
            import graph_evolution
            _graph_evo_module = graph_evolution
            _log("Graph evolution engine loaded")
        except Exception as e:
            _log(f"Graph evolution load failed: {e}")
    return _graph_evo_module

def _graph_evo_cluster_context(params):
    mod = _graph_evo_lazy()
    if mod is None:
        return {"ok": False, "error": "Graph evolution not available"}
    return mod.handle_cluster_context(params)

def _graph_evo_cluster_lookup(params):
    mod = _graph_evo_lazy()
    if mod is None:
        return {"ok": False, "error": "Graph evolution not available"}
    return mod.handle_cluster_for_entity(params)

def _graph_evo_stats(params):
    mod = _graph_evo_lazy()
    if mod is None:
        return {"ok": False, "error": "Graph evolution not available"}
    return mod.handle_graph_stats(params)


# ── Episodic Memory handler shims (Phase 7B) ──────────────────────────────────

_episodic_module = None

def _episodic_lazy():
    global _episodic_module
    if _episodic_module is None:
        try:
            import episodic_memory
            _episodic_module = episodic_memory
            _log("Episodic memory engine loaded")
        except Exception as e:
            _log(f"Episodic memory load failed: {e}")
    return _episodic_module

def _episodic_search(params):
    mod = _episodic_lazy()
    if mod is None:
        return {"ok": False, "error": "Episodic memory not available"}
    return mod.handle_episode_search(params)

def _episodic_context(params):
    mod = _episodic_lazy()
    if mod is None:
        return {"ok": False, "error": "Episodic memory not available"}
    return mod.handle_episode_context(params)

def _episodic_lessons(params):
    mod = _episodic_lazy()
    if mod is None:
        return {"ok": False, "error": "Episodic memory not available"}
    return mod.handle_recent_lessons(params)

def _episodic_stats(params):
    mod = _episodic_lazy()
    if mod is None:
        return {"ok": False, "error": "Episodic memory not available"}
    return mod.handle_episode_stats(params)


# ── Mental Models handler shims (Phase 7C) ────────────────────────────────────

_mental_module = None

def _mental_lazy():
    global _mental_module
    if _mental_module is None:
        try:
            import mental_models
            _mental_module = mental_models
            _log("Mental models engine loaded")
        except Exception as e:
            _log(f"Mental models load failed: {e}")
    return _mental_module

def _mental_predict(params):
    mod = _mental_lazy()
    if mod is None:
        return {"ok": False, "error": "Mental models not available"}
    return mod.handle_predict(params)

def _mental_context(params):
    mod = _mental_lazy()
    if mod is None:
        return {"ok": False, "error": "Mental models not available"}
    return mod.handle_predictive_context(params)

def _mental_entity_model(params):
    mod = _mental_lazy()
    if mod is None:
        return {"ok": False, "error": "Mental models not available"}
    return mod.handle_entity_model(params)

def _mental_stats(params):
    mod = _mental_lazy()
    if mod is None:
        return {"ok": False, "error": "Mental models not available"}
    return mod.handle_model_stats(params)


HANDLERS = {
    "search": handle_search,
    "add": handle_add,
    "get_all": handle_get_all,
    "delete": handle_delete,
    "delete_all": handle_delete_all,
    "status": handle_status,
    # Topic handlers
    "search_topics": handle_search_topics,
    "get_topic_context": handle_get_topic_context,
    "upsert_topic": handle_upsert_topic,
    "add_fact": handle_add_fact,
    "list_topics": handle_list_topics,
    "list_open_loops": handle_list_open_loops,
    "topic_stats": handle_topic_stats,
    # ML-0: Outcome logging
    "log_outcome": handle_log_outcome,
    "ml_stats": handle_ml_stats,
    "ml_label": handle_ml_label,
    # CG-0: Confidence gate
    "gate_stats": handle_gate_stats,
    "gate_record_outcome": handle_gate_record_outcome,
    # LCM: Lossless Context Management
    "lcm_search": _lcm_search,
    "lcm_context": _lcm_context,
    "lcm_stats": _lcm_stats,
    "lcm_pressure": _lcm_pressure,
    "lcm_grep": _lcm_grep,
    "lcm_describe": _lcm_describe,
    "lcm_recall": _lcm_recall,
    # Phase 7A: Graph Evolution
    "cluster_context": _graph_evo_cluster_context,
    "cluster_lookup": _graph_evo_cluster_lookup,
    "graph_stats": _graph_evo_stats,
    # Phase 7B: Episodic Memory
    "episode_search": _episodic_search,
    "episode_context": _episodic_context,
    "episode_lessons": _episodic_lessons,
    "episode_stats": _episodic_stats,
    # Phase 7C: Mental Models + Prediction
    "predict": _mental_predict,
    "predictive_context": _mental_context,
    "entity_model": _mental_entity_model,
    "model_stats": _mental_stats,
}


# ===============================================================================
# Socket server
# ===============================================================================

class MemoRequestHandler(socketserver.StreamRequestHandler):
    """Handle one JSON-line request per connection."""

    def handle(self):
        global _request_count
        try:
            raw = self.rfile.readline(MAX_REQUEST_SIZE)
            if not raw:
                return

            req = json.loads(raw)
            method = req.get("method", "")
            params = req.get("params", {})

            handler = HANDLERS.get(method)
            if handler is None:
                resp = {"ok": False, "error": f"Unknown method: {method}"}
            else:
                t0 = time.time()
                resp = handler(params)
                elapsed_ms = (time.time() - t0) * 1000
                resp["elapsed_ms"] = round(elapsed_ms, 1)
                _request_count += 1

            self.wfile.write(json.dumps(resp).encode() + b"\n")
            self.wfile.flush()

        except json.JSONDecodeError as e:
            resp = {"ok": False, "error": f"Invalid JSON: {e}"}
            self.wfile.write(json.dumps(resp).encode() + b"\n")
            self.wfile.flush()
        except Exception as e:
            _log(f"Handler error: {e}\n{traceback.format_exc()}")
            try:
                resp = {"ok": False, "error": str(e)}
                self.wfile.write(json.dumps(resp).encode() + b"\n")
                self.wfile.flush()
            except Exception:
                pass


class MemoServer(socketserver.UnixStreamServer):
    """Unix socket server with clean shutdown."""
    allow_reuse_address = True

    def server_close(self):
        super().server_close()
        try:
            os.unlink(SOCK_PATH)
        except OSError:
            pass


# ===============================================================================
# Daemon management
# ===============================================================================

def _cleanup(signum=None, frame=None):
    """Clean shutdown handler."""
    _log("Shutting down...")
    try:
        os.unlink(PID_FILE)
    except OSError:
        pass
    try:
        os.unlink(SOCK_PATH)
    except OSError:
        pass
    sys.exit(0)


def _write_pid():
    """Write PID file."""
    Path(PID_FILE).write_text(str(os.getpid()))


def _read_pid():
    """Read PID from file, return int or None."""
    try:
        pid = int(Path(PID_FILE).read_text().strip())
        # Check if process is alive
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return None


def _daemonize():
    """Classic double-fork daemonize."""
    # First fork
    pid = os.fork()
    if pid > 0:
        sys.exit(0)

    os.setsid()

    # Second fork
    pid = os.fork()
    if pid > 0:
        sys.exit(0)

    # Redirect stdio
    sys.stdout.flush()
    sys.stderr.flush()

    log_fd = open(LOG_FILE, "a")
    os.dup2(log_fd.fileno(), sys.stdout.fileno())
    os.dup2(log_fd.fileno(), sys.stderr.fileno())


def cmd_start(foreground=False):
    """Start the daemon."""
    existing_pid = _read_pid()
    if existing_pid:
        print(f"Daemon already running (PID {existing_pid})")
        return 1

    # Remove stale socket
    try:
        os.unlink(SOCK_PATH)
    except OSError:
        pass

    if not foreground:
        print("Starting clyde-memo daemon...")
        _daemonize()

    # Signal handlers
    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)

    _write_pid()
    _log(f"Daemon started (PID {os.getpid()})")

    # Initialize Memory (the expensive part -- do it once)
    try:
        init_memory()
    except Exception as e:
        _log(f"FATAL: Failed to initialize Memory: {e}\n{traceback.format_exc()}")
        _cleanup()
        return 1

    # Start serving
    server = MemoServer(SOCK_PATH, MemoRequestHandler)
    # Make socket world-readable/writable so non-root can use it
    os.chmod(SOCK_PATH, 0o777)
    _log(f"Listening on {SOCK_PATH}")

    try:
        server.serve_forever()
    except Exception as e:
        _log(f"Server error: {e}")
    finally:
        server.server_close()
        _cleanup()


def cmd_stop():
    """Stop the daemon."""
    pid = _read_pid()
    if pid is None:
        print("Daemon is not running")
        return 1

    print(f"Stopping daemon (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait up to 5s for clean exit
        for _ in range(50):
            try:
                os.kill(pid, 0)
                time.sleep(0.1)
            except ProcessLookupError:
                break
        print("Daemon stopped.")
    except Exception as e:
        print(f"Error stopping daemon: {e}")
        return 1
    return 0


def cmd_status_daemon():
    """Check if daemon is running."""
    pid = _read_pid()
    if pid:
        print(f"Daemon is running (PID {pid})")
        print(f"  Socket: {SOCK_PATH}")
        print(f"  Log: {LOG_FILE}")
        return 0
    else:
        print("Daemon is not running")
        return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} start|stop|status|foreground")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "start":
        sys.exit(cmd_start(foreground=False) or 0)
    elif cmd == "foreground":
        sys.exit(cmd_start(foreground=True) or 0)
    elif cmd == "stop":
        sys.exit(cmd_stop())
    elif cmd == "status":
        sys.exit(cmd_status_daemon())
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
