#!/usr/bin/env python3
"""
salience_engine.py -- Upgraded Scoring Engine for OpenClaw.

Phase 2 of the Cognitive Architecture. Replaces the flat weighted scoring
in topic_scorer.py with a richer salience model incorporating goal proximity,
open question boost, narrative position, and working mode matching.

Salience weights:
    Semantic similarity: 0.30  (floor at 0.15)
    Graph walk:          0.10  (weighted BFS through entity graph)
    Recency:             0.15
    Goal proximity:      0.15
    Open question boost: 0.10
    Narrative position:  0.05  (placeholder -- returns 0.5, Phase 3)
    Working mode:        0.10
    Frequency:           0.05

Usage:
    python3 salience_engine.py "query text"
    python3 salience_engine.py "query text" --limit 10
"""

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

# ═══════════════════════════════════════════════════════════════════════════════
# Weights
# ═══════════════════════════════════════════════════════════════════════════════

W_SEMANTIC   = 0.30
W_GRAPH_WALK = 0.10
W_RECENCY    = 0.15
W_GOAL_PROX  = 0.15
W_OQ_BOOST   = 0.10
W_NARRATIVE   = 0.05
W_WORKING    = 0.10
W_FREQUENCY  = 0.05

SEMANTIC_FLOOR = 0.15  # Minimum semantic contribution if score > 0

# Named weight profiles for A/B testing
WEIGHT_PROFILES = {
    "default": {
        "semantic": 0.30, "graph_walk": 0.10, "recency": 0.15, "goal_prox": 0.15,
        "oq_boost": 0.10, "narrative": 0.05, "working_mode": 0.10, "frequency": 0.05,
    },
    "semantic-heavy": {
        "semantic": 0.45, "graph_walk": 0.05, "recency": 0.10, "goal_prox": 0.10,
        "oq_boost": 0.08, "narrative": 0.08, "working_mode": 0.08, "frequency": 0.06,
    },
    "recency-heavy": {
        "semantic": 0.20, "graph_walk": 0.10, "recency": 0.30, "goal_prox": 0.10,
        "oq_boost": 0.10, "narrative": 0.05, "working_mode": 0.10, "frequency": 0.05,
    },
    "goal-focused": {
        "semantic": 0.20, "graph_walk": 0.10, "recency": 0.10, "goal_prox": 0.30,
        "oq_boost": 0.15, "narrative": 0.05, "working_mode": 0.05, "frequency": 0.05,
    },
    "graph-heavy": {
        "semantic": 0.20, "graph_walk": 0.25, "recency": 0.10, "goal_prox": 0.10,
        "oq_boost": 0.10, "narrative": 0.05, "working_mode": 0.10, "frequency": 0.10,
    },
    "balanced": {
        "semantic": 0.20, "graph_walk": 0.15, "recency": 0.15, "goal_prox": 0.15,
        "oq_boost": 0.10, "narrative": 0.05, "working_mode": 0.15, "frequency": 0.05,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# PG helper
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Component scorers
# ═══════════════════════════════════════════════════════════════════════════════

def get_working_mode(query):
    """
    Determine what the user is working on using context_gate.classify_topic.
    Returns a topic string like 'home_automation', 'networking', etc.
    """
    try:
        from context_gate import classify_topic
        return classify_topic(query)
    except ImportError:
        return "general"


def get_oq_boost(query, mem_text):
    """
    Check if the memory text matches any open question.
    Returns 1.0 if there's a match, 0.0 otherwise.

    Matching: word overlap between (query + memory text) and open question text.
    A match requires at least 2 shared meaningful words with any open question.
    """
    try:
        from goal_tracker import get_open_questions, _safe
    except ImportError:
        return 0.0

    questions = get_open_questions(limit=20)
    if not questions:
        return 0.0

    # Build combined text from query + memory
    combined = (query + " " + mem_text).lower()
    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all",
        "can", "had", "her", "was", "one", "our", "out", "has",
        "his", "how", "its", "may", "new", "now", "old", "see",
        "way", "who", "did", "get", "let", "say", "she", "too",
        "use", "this", "that", "with", "have", "from", "they",
        "been", "some", "what", "when", "will", "more", "into",
        "also", "than", "them", "very", "just", "about", "which",
        "their", "there", "would", "could", "should", "where",
    }
    combined_words = set(
        w for w in re.findall(r'[a-z]{3,}', combined)
        if w not in stopwords
    )

    for q in questions:
        q_text = (q.get("question", "") + " " + (q.get("context", "") or "")).lower()
        q_words = set(
            w for w in re.findall(r'[a-z]{3,}', q_text)
            if w not in stopwords
        )
        overlap = combined_words & q_words
        if len(overlap) >= 2:
            return 1.0

    return 0.0


def get_recency_score(mem_id):
    """
    Get recency score for a memory from keyscores table.
    Falls back to 0.3 if not found.
    """
    if not mem_id:
        return 0.3

    result = db.pg_query(
        "SELECT k.recency_score FROM keyscores k "
        "JOIN memories m ON k.memory_id = m.id "
        "WHERE m.qdrant_point_id LIKE %s LIMIT 1;",
        (mem_id + '%',)
    )
    if result:
        try:
            return float(result.strip())
        except ValueError:
            pass
    return 0.3


def get_frequency_score(mem_id):
    """
    Get frequency score for a memory from keyscores table.
    Falls back to 0.1 if not found.
    """
    if not mem_id:
        return 0.1

    result = db.pg_query(
        "SELECT k.frequency_score FROM keyscores k "
        "JOIN memories m ON k.memory_id = m.id "
        "WHERE m.qdrant_point_id LIKE %s LIMIT 1;",
        (mem_id + '%',)
    )
    if result:
        try:
            return float(result.strip())
        except ValueError:
            pass
    return 0.1


def get_narrative_position(query, mem_text):
    """
    Score narrative relevance for a memory based on active narratives.
    Uses narrative_engine from Phase 3 to check if the query/memory
    relates to an active narrative arc and returns its position weight.
    """
    try:
        from narrative_engine import get_narrative_position as _narr_pos
        score = _narr_pos(query)
        return score if score > 0 else 0.5
    except Exception:
        return 0.5


def get_working_mode_match(query, mem_text):
    """
    Score whether memory topic matches the current working mode.
    Returns 1.0 if same topic, 0.0 otherwise.
    """
    try:
        from context_gate import classify_topic
        query_topic = classify_topic(query)
        mem_topic = classify_topic(mem_text)
        if query_topic == mem_topic and query_topic != "general":
            return 1.0
        elif query_topic == "general" or mem_topic == "general":
            return 0.3  # Partial credit for general topics
        return 0.0
    except ImportError:
        return 0.3


# ═══════════════════════════════════════════════════════════════════════════════
# Entity boost (try to import, fallback gracefully)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_graph_walk_score(walked_entities: dict, mem_id: str) -> float:
    """
    Check if a memory's entities overlap with the graph walk results.
    Returns 0.0-0.5 based on the max walk weight among matched entities.
    """
    if not walked_entities or not mem_id:
        return 0.0
    try:
        from graph_walk import _get_memory_entity_ids, MAX_GRAPH_SCORE
        entity_ids = _get_memory_entity_ids(mem_id)
        if not entity_ids:
            return 0.0
        max_weight = 0.0
        for eid in entity_ids:
            if eid in walked_entities:
                max_weight = max(max_weight, walked_entities[eid]["weight"])
        return min(MAX_GRAPH_SCORE, max_weight)
    except (ImportError, Exception):
        return 0.0


def _get_entity_boost(query, mem_id):
    """Get entity boost score, with graceful fallback."""
    try:
        from entity_boost import get_boost
        return get_boost(query, mem_id[:8] if mem_id else "")
    except (ImportError, Exception):
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# FTS search (try to import from topic_scorer, fallback to direct PG)
# ═══════════════════════════════════════════════════════════════════════════════

def _fts_search_direct(query, limit=10):
    """
    Direct FTS search via PG against memories/facts.
    Fallback when topic_scorer is not available.
    Returns list of (qdrant_point_id, rank) tuples.
    """
    # Build tsquery from words
    words = re.findall(r'[a-zA-Z]{3,}', query)
    if not words:
        return []

    ts_query = " | ".join(words[:8])  # OR-based matching

    # Try searching memories.summary with plainto_tsquery
    result = db.pg_query(
        "SELECT m.qdrant_point_id, "
        "ts_rank(to_tsvector('english', COALESCE(m.summary, '')), plainto_tsquery('english', %s)) as rank "
        "FROM memories m "
        "WHERE to_tsvector('english', COALESCE(m.summary, '')) @@ plainto_tsquery('english', %s) "
        "AND m.is_deprecated = FALSE "
        "ORDER BY rank DESC LIMIT %s;",
        (ts_query, ts_query, int(limit))
    )
    hits = []
    if result:
        for line in result.split("\n"):
            if "|" in line:
                parts = line.split("|")
                if len(parts) >= 2:
                    try:
                        hits.append((parts[0], float(parts[1])))
                    except ValueError:
                        pass
    return hits


# ═══════════════════════════════════════════════════════════════════════════════
# Main salience scorer
# ═══════════════════════════════════════════════════════════════════════════════

def salience_score(query, candidates, limit=5, weights=None):
    """
    Score candidates using the full salience model.

    Args:
        query: the search query text
        candidates: list of dicts from Mem0 search results, each having:
            - 'memory': text
            - 'score': semantic similarity (0-1)
            - 'id': qdrant point ID (full UUID)
        limit: max results to return
        weights: optional dict overriding default weights, or a profile name
                 from WEIGHT_PROFILES (e.g. "semantic-heavy")

    Returns:
        list of dicts sorted by final salience score, each containing:
            - 'final': total salience score
            - 'score': original semantic score
            - 'memory': memory text
            - 'mem_id': short ID (first 8 chars)
            - 'id': full UUID
            - 'breakdown': dict of component scores
    """
    if not candidates:
        return []

    # Resolve weight overrides
    if isinstance(weights, str) and weights in WEIGHT_PROFILES:
        w = WEIGHT_PROFILES[weights]
    elif isinstance(weights, dict):
        w = {**WEIGHT_PROFILES["default"], **weights}
    else:
        w = WEIGHT_PROFILES["default"]

    w_sem  = w["semantic"]
    w_gw   = w.get("graph_walk", 0.10)
    w_rec  = w["recency"]
    w_goal = w["goal_prox"]
    w_oq   = w["oq_boost"]
    w_narr = w["narrative"]
    w_wm   = w["working_mode"]
    w_freq = w["frequency"]

    # Pre-compute utility factors for all candidates (one batch query)
    _utility_factors = {}
    try:
        from context_decay import get_utility_factors_batch
        all_ids = [c.get("id", "") for c in candidates if c.get("id")]
        if all_ids:
            _utility_factors = get_utility_factors_batch(all_ids)
    except (ImportError, Exception):
        pass

    # Pre-compute query-level signals (once per query)
    try:
        from goal_tracker import get_goal_proximity
        goal_prox = get_goal_proximity(query)
    except (ImportError, Exception):
        goal_prox = 0.0

    working_mode = get_working_mode(query)

    # Pre-compute graph walk (once per query, shared across candidates)
    _graph_walk_cache = {}
    try:
        from entity_boost import extract_entities_from_text
        from graph_walk import weighted_bfs, expand_via_clusters
        query_entities = list(extract_entities_from_text(query))
        if query_entities:
            walked = weighted_bfs(query_entities[:5], max_hops=2)
            if walked:
                walked = expand_via_clusters(walked)
                _graph_walk_cache = walked
    except (ImportError, Exception):
        query_entities = []

    scored = []
    for c in candidates:
        mem_text = c.get("memory", "")
        raw_semantic = c.get("score", 0.0)
        mem_id = c.get("id", "")
        short_id = mem_id[:8] if mem_id else ""

        # 1. Semantic with floor
        semantic = max(SEMANTIC_FLOOR, raw_semantic) if raw_semantic > 0 else 0.0

        # 2. Graph walk score (per-candidate, uses pre-computed walk)
        gw_score = 0.0
        if _graph_walk_cache and mem_id:
            gw_score = _get_graph_walk_score(_graph_walk_cache, mem_id)

        # For graph-injected candidates, use their existing score as floor
        if c.get("_source") == "graph_walk" and gw_score == 0.0:
            gw_score = min(0.5, c.get("_walk_weight", c.get("score", 0.0)))

        # 3. Recency from keyscores
        recency = get_recency_score(mem_id)

        # 4. Goal proximity (query-level, same for all candidates)
        gp = goal_prox

        # 5. Open question boost (per-candidate)
        oq = get_oq_boost(query, mem_text)

        # 6. Narrative position (placeholder)
        narrative = get_narrative_position(query, mem_text)

        # 7. Working mode match
        wm = get_working_mode_match(query, mem_text)

        # 8. Frequency from keyscores
        frequency = get_frequency_score(mem_id)

        # Weighted sum
        final = (
            w_sem  * semantic +
            w_gw   * gw_score +
            w_rec  * recency +
            w_goal * gp +
            w_oq   * oq +
            w_narr * narrative +
            w_wm   * wm +
            w_freq * frequency
        )

        # Entity boost as additive bonus (not part of the weighted formula)
        entity_boost = _get_entity_boost(query, mem_id)
        final += entity_boost

        # Utility factor: soft multiplier from context_decay loop
        # Ranges 0.5 (heavily decayed) to 1.2 (consistently useful), neutral = 1.0
        util_factor = _utility_factors.get(mem_id, 1.0) if _utility_factors else 1.0
        final *= util_factor

        scored.append({
            "final": round(final, 4),
            "score": round(raw_semantic, 4),
            "memory": mem_text,
            "mem_id": short_id,
            "id": mem_id,
            "breakdown": {
                "semantic": round(semantic, 4),
                "graph_walk": round(gw_score, 4),
                "recency": round(recency, 4),
                "goal_prox": round(gp, 4),
                "oq_boost": round(oq, 4),
                "narrative": round(narrative, 4),
                "working_mode": round(wm, 4),
                "frequency": round(frequency, 4),
                "entity_boost": round(entity_boost, 4),
                "utility_factor": round(util_factor, 4),
            },
        })

    # Sort by final score descending, return top N
    scored.sort(key=lambda x: x["final"], reverse=True)
    return scored[:limit]


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RED = "\033[91m"
RESET = "\033[0m"


def _cli_search(query, limit=5, weights=None):
    """Search using Mem0 and score with salience engine."""

    # Try daemon socket first for raw semantic results
    try:
        from openclaw_memo import _daemon_request, CLYDE_USER
        dresp = _daemon_request("search", {
            "query": query, "user_id": CLYDE_USER,
            "limit": max(limit * 2, 10), "skip_gate": True,
        })
        if dresp and dresp.get("results"):
            # Daemon returns pre-scored results; we need raw semantic candidates
            # Re-fetch via Mem0 for raw scores
            pass
    except ImportError:
        dresp = None

    # Try direct Mem0 for raw semantic results
    candidates = []
    try:
        from openclaw_memo import get_memory, CLYDE_USER
        m = get_memory()
        raw = m.search(query, user_id=CLYDE_USER, limit=max(limit * 2, 10))
        candidates = raw.get("results", [])
    except Exception as e:
        print(f"  {RED}Mem0 unavailable: {e}{RESET}")
        # Fall back to daemon results if available
        if dresp and dresp.get("results"):
            # Convert daemon format to candidates
            for r in dresp["results"]:
                candidates.append({
                    "memory": r.get("memory", ""),
                    "score": r.get("score", r.get("final", 0.0)),
                    "id": r.get("id", r.get("mem_id", "")),
                })

    if not candidates:
        print(f"  {YELLOW}No results found{RESET}")
        return []

    return salience_score(query, candidates, limit=limit, weights=weights)


if __name__ == "__main__":
    import argparse

    profiles = list(WEIGHT_PROFILES.keys())
    parser = argparse.ArgumentParser(
        description="OpenClaw Salience Engine (Phase 2)"
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Max results")
    parser.add_argument("--raw", action="store_true",
                        help="Show raw JSON output")
    parser.add_argument("--profile", choices=profiles, default="default",
                        help=f"Weight profile ({', '.join(profiles)})")
    parser.add_argument("--compare", action="store_true",
                        help="Run all profiles and compare ranking differences")

    args = parser.parse_args()

    def _print_results(results, profile_name, elapsed_ms):
        """Pretty-print scored results."""
        w = WEIGHT_PROFILES[profile_name]
        print(f"\n  {BOLD}[{profile_name}] \"{args.query}\"{RESET}")
        print(f"  {DIM}({len(results)} results in {elapsed_ms:.0f}ms){RESET}")
        print(f"  {DIM}sem={w['semantic']}  gw={w.get('graph_walk', 0.10)}  rec={w['recency']}  goal={w['goal_prox']}  "
              f"oq={w['oq_boost']}  narr={w['narrative']}  wm={w['working_mode']}  "
              f"freq={w['frequency']}{RESET}\n")

        print(f"  {'#':>2}  {'Final':>6}  {'Sem':>5}  {'GW':>5}  {'Rec':>5}  {'Goal':>5}  "
              f"{'OQ':>4}  {'Narr':>5}  {'WM':>4}  {'Freq':>5}  {'Ent':>4}  Memory")
        print(f"  {'-' * 115}")

        for i, r in enumerate(results, 1):
            bd = r["breakdown"]
            mem_preview = r["memory"][:40]
            print(
                f"  {i:2d}  {r['final']:.4f}  "
                f"{bd['semantic']:.3f}  "
                f"{bd.get('graph_walk', 0):.3f}  "
                f"{bd['recency']:.3f}  "
                f"{bd['goal_prox']:.3f}  "
                f"{bd['oq_boost']:.2f}  "
                f"{bd['narrative']:.3f}  "
                f"{bd['working_mode']:.2f}  "
                f"{bd['frequency']:.3f}  "
                f"{bd['entity_boost']:.2f}  "
                f"{mem_preview}  {DIM}({r['mem_id']}){RESET}"
            )

    if args.compare:
        # Run all profiles, show side-by-side ranking comparison
        print(f"\n  {BOLD}{'='*60}")
        print(f"  A/B Weight Comparison: \"{args.query}\"")
        print(f"  {'='*60}{RESET}")

        all_rankings = {}
        for pname in profiles:
            t0 = time.time()
            results = _cli_search(args.query, limit=args.limit, weights=pname)
            elapsed = (time.time() - t0) * 1000
            if results:
                _print_results(results, pname, elapsed)
                all_rankings[pname] = [r["mem_id"] for r in results]

        # Summary: show ranking differences
        if len(all_rankings) > 1:
            print(f"\n  {BOLD}Ranking comparison (by mem_id position):{RESET}")
            ref_name = "default"
            ref_ranking = all_rankings.get(ref_name, [])
            for pname, ranking in all_rankings.items():
                if pname == ref_name:
                    continue
                moves = []
                for i, mid in enumerate(ranking):
                    if mid in ref_ranking:
                        old_pos = ref_ranking.index(mid) + 1
                        new_pos = i + 1
                        if old_pos != new_pos:
                            direction = "↑" if new_pos < old_pos else "↓"
                            moves.append(f"{mid}: {old_pos}→{new_pos}{direction}")
                    else:
                        moves.append(f"{mid}: NEW")
                if moves:
                    print(f"    {CYAN}{pname}{RESET} vs default: {', '.join(moves)}")
                else:
                    print(f"    {CYAN}{pname}{RESET} vs default: identical ranking")
        print()
        sys.exit(0)

    # Single profile run
    t0 = time.time()
    results = _cli_search(args.query, limit=args.limit, weights=args.profile)
    elapsed = (time.time() - t0) * 1000

    if args.raw:
        print(json.dumps(results, indent=2))
        sys.exit(0)

    if not results:
        sys.exit(0)

    _print_results(results, args.profile, elapsed)
    print()
