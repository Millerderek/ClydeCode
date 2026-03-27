#!/usr/bin/env python3
"""
abstraction_engine.py — Pattern Recognition and Learning

Phase 5 of OpenClaw Cognitive Architecture.

Detects repeated patterns across memories and extracts abstract rules/lessons
that Luther can apply to new situations. Implements a maturity ladder that
mirrors soul.md preference maturation:

  observation → leaning → soft_pattern → stable_pattern → principle

Usage:
    python3 abstraction_engine.py analyze
    python3 abstraction_engine.py list [--type workflow] [--maturity stable_pattern]
    python3 abstraction_engine.py query "search query"
    python3 abstraction_engine.py context "query"
"""

import json
import os
import sys
import subprocess
import re
import argparse
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

VALID_PATTERN_TYPES = {
    "workflow", "preference", "failure_mode", "success_pattern", "heuristic"
}
MATURITY_LADDER = ["observation", "leaning", "soft_pattern", "stable_pattern", "principle"]
MODEL = "anthropic/claude-haiku-4.5"

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_openrouter_key():
    """Get OpenRouter API key from environment or vault."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    key_file = os.environ.get("OPENROUTER_KEY_FILE",
                               os.path.expanduser("~/APIKeys/openrouter.env"))
    try:
        with open(key_file) as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.strip().split("=", 1)[1]
    except Exception:
        pass
    return ""


def _pg(sql, params=None):
    """Run a SQL statement via db module. Returns list of rows (pipe-delimited)."""
    result = db.pg_query(sql, params)
    if not result:
        return []
    return [ln for ln in result.split("\n") if ln]


def _pg_single(sql, params=None):
    """Run SQL, return single scalar value or None."""
    rows = _pg(sql, params)
    return rows[0] if rows else None


def _llm_call(prompt, max_tokens=500):
    """Call OpenRouter Haiku for pattern extraction."""
    key = _get_openrouter_key()
    if not key:
        print("[llm] No OpenRouter API key found", file=sys.stderr)
        return ""

    payload = json.dumps({
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    })

    cmd = [
        "curl", "-s", "-X", "POST",
        "https://openrouter.ai/api/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {key}",
        "-d", payload,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        data = json.loads(result.stdout)
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[llm error] {e}", file=sys.stderr)
        return ""


def _extract_json(text):
    """Extract JSON from LLM response, handling markdown fences."""
    if not text:
        return text
    # Strip markdown code fences
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()



# ═══════════════════════════════════════════════════════════════════════════════
# Pattern Management
# ═══════════════════════════════════════════════════════════════════════════════

def add_pattern(pattern_type, title, description, evidence_memories=None,
                confidence=0.5, related_entities=None, tags=None):
    """Insert a new learned pattern. Returns pattern id or None on conflict."""
    if pattern_type not in VALID_PATTERN_TYPES:
        print(f"[warn] Invalid pattern_type: {pattern_type}", file=sys.stderr)
        return None

    evidence_memories = evidence_memories or []
    related_entities = related_entities or []
    tags = tags or []

    evidence_count = max(1, len(evidence_memories))

    row = _pg_single(
        "INSERT INTO learned_patterns "
        "  (pattern_type, title, description, evidence_count, confidence, "
        "   related_entities, tags) "
        "VALUES (%s, %s, %s, %s, %s, %s::text[], %s::text[]) "
        "ON CONFLICT (title) DO UPDATE SET "
        "  evidence_count = learned_patterns.evidence_count + %s, "
        "  confidence = LEAST(1.0, learned_patterns.confidence + 0.05), "
        "  last_confirmed = NOW(), "
        "  description = CASE "
        "    WHEN length(%s) > length(learned_patterns.description) "
        "    THEN %s "
        "    ELSE learned_patterns.description "
        "  END "
        "RETURNING id;",
        (pattern_type, title, description, evidence_count, confidence,
         related_entities, tags, evidence_count, description, description)
    )
    if not row:
        return None

    pattern_id = int(row)

    # Link evidence memories
    for mem_id in evidence_memories:
        _pg(
            "INSERT INTO pattern_evidence (pattern_id, memory_id, evidence_type, summary) "
            "VALUES (%s, %s, 'supports', 'Auto-detected during analysis') "
            "ON CONFLICT DO NOTHING;",
            (pattern_id, mem_id)
        )

    # Auto-mature based on new evidence count
    mature_pattern(pattern_id)

    return pattern_id


def confirm_pattern(pattern_id, memory_id=None, summary="Confirmed"):
    """Add supporting evidence, increment count, update confidence and maturity."""
    _pg(
        "INSERT INTO pattern_evidence (pattern_id, memory_id, evidence_type, summary) "
        "VALUES (%s, %s, 'supports', %s);",
        (pattern_id, memory_id, summary)
    )
    _pg(
        "UPDATE learned_patterns SET "
        "  evidence_count = evidence_count + 1, "
        "  confidence = LEAST(1.0, confidence + 0.03), "
        "  last_confirmed = NOW() "
        "WHERE id = %s;",
        (pattern_id,)
    )
    mature_pattern(pattern_id)


def challenge_pattern(pattern_id, memory_id=None, summary="Challenged"):
    """Add challenging evidence, increment challenge count, potentially demote."""
    _pg(
        "INSERT INTO pattern_evidence (pattern_id, memory_id, evidence_type, summary) "
        "VALUES (%s, %s, 'challenges', %s);",
        (pattern_id, memory_id, summary)
    )
    _pg(
        "UPDATE learned_patterns SET "
        "  challenge_count = challenge_count + 1, "
        "  confidence = GREATEST(0.0, confidence - 0.05), "
        "  last_challenged = NOW() "
        "WHERE id = %s;",
        (pattern_id,)
    )

    # Check for demotion: if challenge rate > 40%, demote one level
    rows = _pg(
        "SELECT evidence_count, challenge_count, maturity "
        "FROM learned_patterns WHERE id = %s;",
        (pattern_id,)
    )
    if rows:
        parts = rows[0].split("|")
        ev, ch, mat = int(parts[0]), int(parts[1]), parts[2]
        if ev > 0 and ch / ev > 0.4:
            idx = MATURITY_LADDER.index(mat) if mat in MATURITY_LADDER else 0
            if idx > 0:
                new_mat = MATURITY_LADDER[idx - 1]
                _pg(
                    "UPDATE learned_patterns SET maturity = %s WHERE id = %s;",
                    (new_mat, pattern_id)
                )
                print(f"  Pattern {pattern_id} demoted to {new_mat} (challenge rate {ch}/{ev})")


def mature_pattern(pattern_id):
    """Promote pattern based on evidence_count vs challenge_count.

    Maturity ladder:
      observation (1-2) → leaning (3-4) → soft_pattern (5-7)
      → stable_pattern (8-12) → principle (13+ with <20% challenge rate)
    """
    rows = _pg(
        "SELECT evidence_count, challenge_count, maturity "
        "FROM learned_patterns WHERE id = %s;",
        (pattern_id,)
    )
    if not rows:
        return

    parts = rows[0].split("|")
    ev, ch, mat = int(parts[0]), int(parts[1]), parts[2]
    challenge_rate = ch / ev if ev > 0 else 0

    # Determine target maturity
    if ev >= 13 and challenge_rate < 0.2:
        target = "principle"
    elif ev >= 8:
        target = "stable_pattern"
    elif ev >= 5:
        target = "soft_pattern"
    elif ev >= 3:
        target = "leaning"
    else:
        target = "observation"

    # Only promote, never demote via this function (challenge_pattern handles demotion)
    cur_idx = MATURITY_LADDER.index(mat) if mat in MATURITY_LADDER else 0
    tgt_idx = MATURITY_LADDER.index(target)
    if tgt_idx > cur_idx:
        _pg(
            "UPDATE learned_patterns SET maturity = %s WHERE id = %s;",
            (target, pattern_id)
        )
        print(f"  Pattern {pattern_id} matured to {target} (evidence={ev}, challenges={ch})")


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def get_relevant_patterns(query, limit=3):
    """Find patterns relevant to a query via FTS + entity overlap."""
    # Build tsquery from words
    words = re.findall(r"[a-zA-Z0-9_]+", query)
    if not words:
        return []
    ts_terms = " | ".join(words[:10])
    words_lower = [w.lower() for w in words[:10]]

    rows = _pg(
        "SELECT id, pattern_type, title, description, confidence, maturity, "
        "       evidence_count, challenge_count, related_entities, tags "
        "FROM learned_patterns "
        "WHERE is_active = TRUE "
        "  AND ( "
        "    to_tsvector('english', title || ' ' || description) "
        "      @@ to_tsquery('english', %s) "
        "    OR EXISTS ( "
        "      SELECT 1 FROM unnest(related_entities) AS e "
        "      WHERE lower(e) = ANY(%s) "
        "    ) "
        "  ) "
        "ORDER BY confidence DESC, evidence_count DESC "
        "LIMIT %s;",
        (ts_terms, words_lower, limit)
    )

    return _parse_pattern_rows(rows)


def get_all_patterns(pattern_type=None, maturity=None):
    """Get all patterns, optionally filtered."""
    clauses = ["is_active = TRUE"]
    params = []
    if pattern_type:
        clauses.append("pattern_type = %s")
        params.append(pattern_type)
    if maturity:
        clauses.append("maturity = %s")
        params.append(maturity)

    where = " AND ".join(clauses)
    rows = _pg(
        "SELECT id, pattern_type, title, description, confidence, maturity, "
        "       evidence_count, challenge_count, related_entities, tags "
        f"FROM learned_patterns WHERE {where} "
        "ORDER BY confidence DESC, evidence_count DESC;",
        tuple(params) if params else None
    )

    return _parse_pattern_rows(rows)


def _parse_pattern_rows(rows):
    """Parse pipe-delimited pattern rows into dicts."""
    patterns = []
    for row in rows:
        parts = row.split("|")
        if len(parts) < 10:
            continue
        patterns.append({
            "id": int(parts[0]),
            "pattern_type": parts[1],
            "title": parts[2],
            "description": parts[3],
            "confidence": float(parts[4]),
            "maturity": parts[5],
            "evidence_count": int(parts[6]),
            "challenge_count": int(parts[7]),
            "related_entities": parts[8].strip("{}").split(",") if parts[8].strip("{}") else [],
            "tags": parts[9].strip("{}").split(",") if parts[9].strip("{}") else [],
        })
    return patterns


def build_patterns_context(query, limit=3):
    """Build a <LEARNED_PATTERNS> block for prompt injection."""
    patterns = get_relevant_patterns(query, limit=limit)
    if not patterns:
        return ""

    lines = ["<LEARNED_PATTERNS>"]
    for p in patterns:
        maturity_marker = f"[{p['maturity']}]"
        conf = f"{p['confidence']:.0%}"
        lines.append(f"- {maturity_marker} {p['title']} (conf={conf}, evidence={p['evidence_count']})")
        lines.append(f"  {p['description']}")
    lines.append("</LEARNED_PATTERNS>")

    return "\n".join(lines)


def get_topic_matched_patterns(topic_tags: list[str], limit: int = 3) -> list[dict]:
    """Find patterns whose tags or entities overlap with the given topic tags.

    This enables topic-aware pattern injection: instead of relying only on
    FTS keyword match against the raw query, we match patterns whose tags
    or related_entities intersect with the conversation's detected topics.
    """
    if not topic_tags:
        return []

    # Normalize tags for matching
    normalized = [t.lower().strip() for t in topic_tags if t.strip()]
    if not normalized:
        return []

    # Build SQL: match patterns where tags or related_entities overlap
    rows = _pg(
        "SELECT id, pattern_type, title, description, confidence, maturity, "
        "       evidence_count, challenge_count, related_entities, tags "
        "FROM learned_patterns "
        "WHERE is_active = TRUE "
        "  AND ( "
        "    EXISTS ( "
        "      SELECT 1 FROM unnest(tags) AS t "
        "      WHERE lower(t) = ANY(%s) "
        "    ) "
        "    OR EXISTS ( "
        "      SELECT 1 FROM unnest(related_entities) AS e "
        "      WHERE lower(e) = ANY(%s) "
        "    ) "
        "  ) "
        "ORDER BY confidence DESC, evidence_count DESC "
        "LIMIT %s;",
        (normalized, normalized, limit)
    )

    return _parse_pattern_rows(rows)


def build_topic_patterns_context(query: str, topic_tags: list[str] | None = None,
                                  limit: int = 3) -> str:
    """Build patterns context using topic-aware matching with FTS fallback.

    1. If topic_tags are provided, try topic-matched patterns first.
    2. Fall back to FTS query match if topic matching returns too few.
    3. Deduplicate by pattern ID.
    """
    seen_ids = set()
    combined = []

    # Phase 1: Topic-matched patterns
    if topic_tags:
        topic_patterns = get_topic_matched_patterns(topic_tags, limit=limit)
        for p in topic_patterns:
            if p["id"] not in seen_ids:
                seen_ids.add(p["id"])
                combined.append(p)

    # Phase 2: FTS fallback for remaining slots
    remaining = limit - len(combined)
    if remaining > 0:
        fts_patterns = get_relevant_patterns(query, limit=remaining + 2)
        for p in fts_patterns:
            if p["id"] not in seen_ids and len(combined) < limit:
                seen_ids.add(p["id"])
                combined.append(p)

    if not combined:
        return ""

    lines = ["<LEARNED_PATTERNS>"]
    for p in combined:
        maturity_marker = f"[{p['maturity']}]"
        conf = f"{p['confidence']:.0%}"
        lines.append(f"- {maturity_marker} {p['title']} (conf={conf}, evidence={p['evidence_count']})")
        lines.append(f"  {p['description']}")
    lines.append("</LEARNED_PATTERNS>")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Pattern Detection
# ═══════════════════════════════════════════════════════════════════════════════

def _get_recent_memories(hours=48):
    """Fetch recent memories with their linked entities."""
    rows = _pg(f"""
        SELECT m.id, m.summary,
               COALESCE(string_agg(DISTINCT e.name, ', '), '') as entities
        FROM memories m
        LEFT JOIN memory_entity_links mel ON mel.memory_id = m.id
        LEFT JOIN entities e ON e.id = mel.entity_id
        WHERE m.created_at > NOW() - INTERVAL '{hours} hours'
          AND m.is_deprecated = FALSE
        GROUP BY m.id, m.summary
        ORDER BY m.created_at DESC;
    """)
    memories = []
    for row in rows:
        parts = row.split("|", 2)
        if len(parts) < 3:
            continue
        memories.append({
            "id": parts[0],
            "summary": parts[1],
            "entities": [e.strip() for e in parts[2].split(",") if e.strip()],
        })
    return memories


def _group_by_entity_overlap(memories, min_shared=2, min_group_size=3):
    """Group memories that share at least min_shared entities."""
    groups = []
    used = set()

    for i, m1 in enumerate(memories):
        if i in used or not m1["entities"]:
            continue
        group = [m1]
        group_entities = set(m1["entities"])

        for j, m2 in enumerate(memories):
            if j <= i or j in used or not m2["entities"]:
                continue
            shared = group_entities & set(m2["entities"])
            if len(shared) >= min_shared:
                group.append(m2)
                group_entities |= set(m2["entities"])
                used.add(j)

        if len(group) >= min_group_size:
            used.add(i)
            groups.append({
                "memories": group,
                "shared_entities": list(group_entities),
            })

    return groups


def detect_workflow_patterns():
    """Find memories describing similar action sequences."""
    print("[detect] Scanning for workflow patterns...")
    memories = _get_recent_memories(hours=168)  # 7 days for workflows
    if len(memories) < 5:
        print(f"  Only {len(memories)} memories with entities, need 5+. Skipping workflows.")
        return []

    groups = _group_by_entity_overlap(memories, min_shared=3, min_group_size=5)
    patterns_found = []

    for grp in groups:
        summaries = "\n".join(f"- {m['summary']}" for m in grp["memories"][:15])
        entities = ", ".join(grp["shared_entities"][:10])

        prompt = f"""Analyze these related memories and identify any repeated workflow pattern
(a sequence of actions that appears multiple times).

Shared entities: {entities}

Memories:
{summaries}

If you find a clear workflow pattern, respond with JSON:
{{"found": true, "title": "short title", "description": "describe the repeated workflow in 1-3 sentences", "tags": ["tag1", "tag2"]}}

If no clear workflow pattern exists, respond with:
{{"found": false}}

JSON only, no other text."""

        resp = _llm_call(prompt)
        try:
            data = json.loads(_extract_json(resp))
            if data.get("found"):
                mem_ids = [m["id"] for m in grp["memories"]]
                pid = add_pattern(
                    pattern_type="workflow",
                    title=data["title"],
                    description=data["description"],
                    evidence_memories=mem_ids,
                    confidence=min(0.4 + len(mem_ids) * 0.05, 0.8),
                    related_entities=grp["shared_entities"],
                    tags=data.get("tags", []),
                )
                if pid:
                    patterns_found.append(pid)
                    print(f"  Workflow pattern: {data['title']} (id={pid})")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [warn] LLM parse error: {e}", file=sys.stderr)

    return patterns_found


def detect_failure_modes():
    """Find memories about errors/fixes with common entities."""
    print("[detect] Scanning for failure modes...")

    # Look for memories whose summaries mention fix, error, bug, broke, fail
    rows = _pg("""
        SELECT m.id, m.summary,
               COALESCE(string_agg(DISTINCT e.name, ', '), '') as entities
        FROM memories m
        LEFT JOIN memory_entity_links mel ON mel.memory_id = m.id
        LEFT JOIN entities e ON e.id = mel.entity_id
        WHERE m.is_deprecated = FALSE
          AND (m.summary ILIKE '%fix%' OR m.summary ILIKE '%error%'
               OR m.summary ILIKE '%bug%' OR m.summary ILIKE '%broke%'
               OR m.summary ILIKE '%fail%' OR m.summary ILIKE '%issue%'
               OR m.summary ILIKE '%crash%' OR m.summary ILIKE '%troubleshoot%')
        GROUP BY m.id, m.summary
        ORDER BY m.created_at DESC
        LIMIT 50;
    """)

    memories = []
    for row in rows:
        parts = row.split("|", 2)
        if len(parts) < 3:
            continue
        memories.append({
            "id": parts[0],
            "summary": parts[1],
            "entities": [e.strip() for e in parts[2].split(",") if e.strip()],
        })

    if len(memories) < 3:
        print(f"  Only {len(memories)} error-related memories. Skipping failure modes.")
        return []

    groups = _group_by_entity_overlap(memories, min_shared=1, min_group_size=3)

    # If no groups from overlap, treat all as one group
    if not groups and len(memories) >= 5:
        groups = [{"memories": memories[:15], "shared_entities": []}]

    patterns_found = []
    for grp in groups:
        summaries = "\n".join(f"- {m['summary']}" for m in grp["memories"][:15])
        entities = ", ".join(grp["shared_entities"][:10]) if grp["shared_entities"] else "various"

        prompt = f"""Analyze these error/fix memories and identify a recurring failure mode pattern.

Related entities: {entities}

Memories:
{summaries}

If you find a repeated failure pattern (same kind of error happening, or same fix being applied), respond with JSON:
{{"found": true, "title": "short title", "description": "what goes wrong and what fixes it, in 1-3 sentences", "tags": ["tag1"]}}

If no clear failure pattern, respond with:
{{"found": false}}

JSON only, no other text."""

        resp = _llm_call(prompt)
        try:
            data = json.loads(_extract_json(resp))
            if data.get("found"):
                mem_ids = [m["id"] for m in grp["memories"]]
                pid = add_pattern(
                    pattern_type="failure_mode",
                    title=data["title"],
                    description=data["description"],
                    evidence_memories=mem_ids,
                    confidence=min(0.3 + len(mem_ids) * 0.05, 0.75),
                    related_entities=grp.get("shared_entities", []),
                    tags=data.get("tags", []),
                )
                if pid:
                    patterns_found.append(pid)
                    print(f"  Failure mode: {data['title']} (id={pid})")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [warn] LLM parse error: {e}", file=sys.stderr)

    return patterns_found


def detect_preferences():
    """Find memories about decisions/choices with repeated patterns."""
    print("[detect] Scanning for preferences...")

    rows = _pg("""
        SELECT m.id, m.summary,
               COALESCE(string_agg(DISTINCT e.name, ', '), '') as entities
        FROM memories m
        LEFT JOIN memory_entity_links mel ON mel.memory_id = m.id
        LEFT JOIN entities e ON e.id = mel.entity_id
        WHERE m.is_deprecated = FALSE
          AND (m.summary ILIKE '%prefer%' OR m.summary ILIKE '%chose%'
               OR m.summary ILIKE '%decided%' OR m.summary ILIKE '%uses %'
               OR m.summary ILIKE '%switched%' OR m.summary ILIKE '%instead of%'
               OR m.summary ILIKE '%rather than%' OR m.summary ILIKE '%always%'
               OR m.summary ILIKE '%configured%' OR m.summary ILIKE '%setup%'
               OR m.summary ILIKE '%approach%')
        GROUP BY m.id, m.summary
        ORDER BY m.created_at DESC
        LIMIT 50;
    """)

    memories = []
    for row in rows:
        parts = row.split("|", 2)
        if len(parts) < 3:
            continue
        memories.append({
            "id": parts[0],
            "summary": parts[1],
            "entities": [e.strip() for e in parts[2].split(",") if e.strip()],
        })

    if len(memories) < 3:
        print(f"  Only {len(memories)} preference-related memories. Skipping.")
        return []

    # For preferences, we send them in a batch to the LLM
    summaries = "\n".join(f"- {m['summary']}" for m in memories[:25])

    prompt = f"""Analyze these memories about decisions, configurations, and choices.
Identify any repeated preferences or decision patterns (things the user consistently chooses).

Memories:
{summaries}

List up to 3 clear preference patterns as JSON array:
[{{"title": "short title", "description": "what they prefer and why, 1-2 sentences", "tags": ["tag1"], "evidence_indices": [0, 3, 7]}}]

evidence_indices = which memories (0-indexed) support this pattern.
If no clear preferences, respond with: []

JSON only, no other text."""

    resp = _llm_call(prompt, max_tokens=800)
    patterns_found = []

    try:
        data = json.loads(_extract_json(resp))
        if not isinstance(data, list):
            return []

        for item in data:
            if not item.get("title"):
                continue
            indices = item.get("evidence_indices", [])
            mem_ids = [memories[i]["id"] for i in indices if i < len(memories)]
            all_entities = []
            for i in indices:
                if i < len(memories):
                    all_entities.extend(memories[i]["entities"])
            unique_entities = list(set(all_entities))

            pid = add_pattern(
                pattern_type="preference",
                title=item["title"],
                description=item["description"],
                evidence_memories=mem_ids if mem_ids else [memories[0]["id"]],
                confidence=min(0.4 + len(mem_ids) * 0.06, 0.8),
                related_entities=unique_entities[:10],
                tags=item.get("tags", []),
            )
            if pid:
                patterns_found.append(pid)
                print(f"  Preference: {item['title']} (id={pid})")

    except (json.JSONDecodeError, KeyError) as e:
        print(f"  [warn] LLM parse error: {e}", file=sys.stderr)

    return patterns_found


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis — Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis():
    """Main analysis entry point. Run periodically (every 6 hours).

    Scans recent memories, groups by entity overlap, calls LLM to extract
    patterns. Stores new patterns or confirms existing ones.
    """
    print("=" * 60)
    print(f"Abstraction Engine — Analysis Run")
    print(f"  Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    total = []

    # Run all detectors
    total.extend(detect_workflow_patterns())
    total.extend(detect_failure_modes())
    total.extend(detect_preferences())

    # Also check if any recent memories confirm existing patterns
    _confirm_existing_patterns()

    print("-" * 60)
    print(f"Analysis complete. {len(total)} new/updated patterns.")

    # Summary of all active patterns
    all_patterns = get_all_patterns()
    by_maturity = {}
    for p in all_patterns:
        by_maturity.setdefault(p["maturity"], []).append(p)

    print(f"Total active patterns: {len(all_patterns)}")
    for mat in MATURITY_LADDER:
        count = len(by_maturity.get(mat, []))
        if count:
            print(f"  {mat}: {count}")

    return {
        "new_patterns": len(total),
        "total_active": len(all_patterns),
        "by_maturity": {k: len(v) for k, v in by_maturity.items()},
    }


def _confirm_existing_patterns():
    """Check recent memories against existing patterns for confirmation."""
    patterns = get_all_patterns()
    if not patterns:
        return

    recent = _get_recent_memories(hours=48)
    if not recent:
        return

    for pat in patterns:
        pat_entities = set(e.strip('"').lower() for e in pat["related_entities"] if e)
        if not pat_entities:
            continue

        for mem in recent:
            mem_entities = set(e.lower() for e in mem["entities"])
            overlap = pat_entities & mem_entities
            if len(overlap) >= 2:
                # Check if this memory is already linked
                existing = _pg(
                    "SELECT id FROM pattern_evidence "
                    "WHERE pattern_id = %s AND memory_id = %s;",
                    (pat['id'], mem["id"])
                )
                if not existing:
                    confirm_pattern(pat["id"], mem["id"],
                                    f"Entity overlap: {', '.join(overlap)}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def cli():
    parser = argparse.ArgumentParser(
        description="Abstraction Engine — Pattern Recognition and Learning"
    )
    sub = parser.add_subparsers(dest="command")

    # analyze
    sub.add_parser("analyze", help="Run full pattern analysis")

    # list
    list_p = sub.add_parser("list", help="List learned patterns")
    list_p.add_argument("--type", dest="ptype", choices=sorted(VALID_PATTERN_TYPES),
                        help="Filter by pattern type")
    list_p.add_argument("--maturity", choices=MATURITY_LADDER,
                        help="Filter by maturity level")

    # query
    q_p = sub.add_parser("query", help="Find relevant patterns")
    q_p.add_argument("search", help="Search query")
    q_p.add_argument("--limit", type=int, default=3)

    # context
    ctx_p = sub.add_parser("context", help="Build context block for prompt injection")
    ctx_p.add_argument("search", help="Query to build context for")
    ctx_p.add_argument("--limit", type=int, default=3)

    args = parser.parse_args()

    if args.command == "analyze":
        result = run_analysis()
        print(f"\n{json.dumps(result, indent=2)}")

    elif args.command == "list":
        patterns = get_all_patterns(pattern_type=args.ptype, maturity=args.maturity)
        if not patterns:
            print("No patterns found.")
            return
        for p in patterns:
            mat_tag = f"[{p['maturity']}]"
            conf = f"{p['confidence']:.0%}"
            chall = f" ({p['challenge_count']} challenges)" if p["challenge_count"] else ""
            print(f"#{p['id']:3d} {mat_tag:18s} {p['pattern_type']:15s} "
                  f"conf={conf:4s} ev={p['evidence_count']:2d}{chall}")
            print(f"     {p['title']}")
            desc = p['description'][:120] + "..." if len(p['description']) > 120 else p['description']
            print(f"     {desc}")
            if p["related_entities"]:
                ents = ", ".join(e.strip('"') for e in p["related_entities"][:5])
                print(f"     entities: {ents}")
            print()

    elif args.command == "query":
        patterns = get_relevant_patterns(args.search, limit=args.limit)
        if not patterns:
            print("No relevant patterns found.")
            return
        for p in patterns:
            print(f"[{p['maturity']}] {p['title']} (conf={p['confidence']:.0%})")
            print(f"  {p['description']}")
            print()

    elif args.command == "context":
        block = build_patterns_context(args.search, limit=args.limit)
        if block:
            print(block)
        else:
            print("No relevant patterns to inject.")

    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
