#!/usr/bin/env python3
"""
graph_extractor.py — Extract entity-relationship triples from memory text.

Called async after memo_daemon add. Uses cheap LLM (Haiku) for structured output.
Upserts entities + relationships into PG graph tables.
Low-confidence triples logged for review, not auto-inserted.

Phase 1 of OpenClaw Cognitive Architecture.
"""

import json
import os
import sys
import time
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

CONFIDENCE_THRESHOLD = 0.7  # Below this, log for review instead of inserting
VALID_PREDICATES = {
    "USES", "CONFIGURED_BY", "DEPENDS_ON", "MIGRATING_FROM",
    "MANAGED_BY", "REPORTS_TO", "INVOLVES", "HAS_QUESTION",
    "PRECEDED_BY", "RUNS_ON", "CONNECTED_TO", "PART_OF",
    "MONITORS", "BACKS_UP", "SERVES", "AUTHENTICATES_WITH",
}
VALID_ENTITY_TYPES = {
    "client", "system", "script", "agent", "infrastructure",
    "person", "goal", "open_question", "narrative_arc",
    "device", "service", "protocol", "general",
}

EXTRACTION_PROMPT = """Extract entity-relationship triples from this memory.
Output JSON only. No preamble.

Entity types: client, system, script, agent, infrastructure, person, device, service, protocol, general
Predicates: USES, CONFIGURED_BY, DEPENDS_ON, MIGRATING_FROM, MANAGED_BY, REPORTS_TO, INVOLVES, RUNS_ON, CONNECTED_TO, PART_OF, MONITORS, BACKS_UP, SERVES, AUTHENTICATES_WITH

Rules:
- Only extract relationships explicitly stated or strongly implied
- Use canonical entity names (e.g. "Operator Connect" not "OC")
- Each triple needs: subject, subject_type, predicate, object, object_type, confidence (0-1)
- Skip trivial relationships (e.g. "user uses computer")
- Confidence < 0.7 means the relationship is inferred, not stated

Memory: "{text}"

Output format:
[
  {{"subject": "...", "subject_type": "...", "predicate": "...", "object": "...", "object_type": "...", "confidence": 0.9}}
]"""


# ═══════════════════════════════════════════════════════════════════════════════
# LLM extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _get_openrouter_key():
    """Get OpenRouter API key from environment or vault."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    # Try loading from file
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


def extract_triples(text: str) -> list:
    """
    Call LLM to extract entity-relationship triples from memory text.
    Returns list of dicts with subject, predicate, object, confidence.
    """
    import urllib.request

    api_key = _get_openrouter_key()
    if not api_key:
        return []

    prompt = EXTRACTION_PROMPT.format(text=text.replace('"', '\\"'))

    payload = json.dumps({
        "model": "anthropic/claude-haiku-4.5",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.0,
    })

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload.encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/openclaw/ClydeMemory",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        content = result["choices"][0]["message"]["content"].strip()

        # Parse JSON from response (handle markdown code blocks)
        if content.startswith("```"):
            content = re.sub(r"^```\w*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

        triples = json.loads(content)
        if not isinstance(triples, list):
            return []

        # Validate and normalize
        valid = []
        for t in triples:
            if not all(k in t for k in ("subject", "predicate", "object")):
                continue
            pred = t["predicate"].upper().replace(" ", "_")
            if pred not in VALID_PREDICATES:
                continue
            t["predicate"] = pred
            t["subject_type"] = t.get("subject_type", "general").lower()
            t["object_type"] = t.get("object_type", "general").lower()
            if t["subject_type"] not in VALID_ENTITY_TYPES:
                t["subject_type"] = "general"
            if t["object_type"] not in VALID_ENTITY_TYPES:
                t["object_type"] = "general"
            t["confidence"] = float(t.get("confidence", 0.8))
            valid.append(t)
        return valid

    except Exception as e:
        print(f"[graph_extractor] LLM extraction failed: {e}", file=sys.stderr)
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# PG upsert
# ═══════════════════════════════════════════════════════════════════════════════

def _run_pg(sql, params=None):
    """Run SQL via db module. Returns pipe-delimited text."""
    result = db.pg_query(sql, params)
    return result if result else ""


def _upsert_entity(name: str, entity_type: str) -> int:
    """Upsert an entity, return its ID. Check aliases first."""
    # Check alias first
    result = _run_pg(
        "SELECT entity_id FROM entity_aliases WHERE alias ILIKE %s LIMIT 1;",
        (name,)
    )
    if result:
        return int(result.split("|")[0])

    # Check exact match
    result = _run_pg(
        "SELECT id FROM entities WHERE name ILIKE %s AND type = %s LIMIT 1;",
        (name, entity_type)
    )
    if result:
        return int(result.split("|")[0])

    # Check fuzzy match (same name, any type)
    result = _run_pg(
        "SELECT id FROM entities WHERE name ILIKE %s LIMIT 1;",
        (name,)
    )
    if result:
        return int(result.split("|")[0])

    # Normalize name before insert
    try:
        sys.path.insert(0, os.path.expanduser("~/openclaw-memory"))
        from entity_normalize import normalize_entity_name
        name = normalize_entity_name(name)
    except ImportError:
        pass

    # Insert new
    result = _run_pg(
        "INSERT INTO entities (name, type) VALUES (%s, %s) "
        "ON CONFLICT (name, type) DO UPDATE SET updated_at = NOW() "
        "RETURNING id;",
        (name, entity_type)
    )
    if result:
        return int(result.split("|")[0])
    return 0


def _upsert_relationship(source_id: int, target_id: int, predicate: str, props: dict = None):
    """Upsert a relationship with evolution tracking (Phase 7A).
    Increments observation_count and refreshes decay on re-observation."""
    try:
        from graph_evolution import reinforce_relationship
        reinforce_relationship(source_id, target_id, predicate, props)
    except ImportError:
        # Fallback to static upsert if graph_evolution not available
        props_json = json.dumps(props or {})
        db.pg_execute(
            "INSERT INTO relationships (source_id, target_id, predicate, properties) "
            "VALUES (%s, %s, %s, %s::jsonb) "
            "ON CONFLICT (source_id, target_id, predicate) DO UPDATE SET "
            "properties = %s::jsonb, created_at = NOW();",
            (source_id, target_id, predicate, props_json, props_json)
        )


def _link_memory_entity(memory_uuid: str, entity_id: int):
    """Link a memory to an entity."""
    db.pg_execute(
        "INSERT INTO memory_entity_links (memory_id, entity_id) "
        "SELECT id, %s FROM memories WHERE qdrant_point_id = %s "
        "ON CONFLICT DO NOTHING;",
        (entity_id, memory_uuid)
    )


def _log_extraction(memory_uuid: str, point_id: str, status: str, count: int, error: str = None):
    """Log extraction result."""
    if error:
        db.pg_execute(
            "INSERT INTO extraction_log (memory_id, qdrant_point_id, status, triples_count, error) "
            "SELECT id, %s, %s, %s, %s FROM memories "
            "WHERE qdrant_point_id = %s LIMIT 1;",
            (point_id, status, count, error, memory_uuid)
        )
    else:
        db.pg_execute(
            "INSERT INTO extraction_log (memory_id, qdrant_point_id, status, triples_count) "
            "SELECT id, %s, %s, %s FROM memories "
            "WHERE qdrant_point_id = %s LIMIT 1;",
            (point_id, status, count, memory_uuid)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Main extraction pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def extract_and_store(text: str, memory_id: str):
    """
    Full pipeline: extract triples from text → upsert to PG graph.

    Args:
        text: memory text
        memory_id: qdrant_point_id (full UUID)
    """
    short_id = memory_id[:8]

    # Skip very short memories
    if len(text) < 20:
        return {"status": "skipped", "reason": "too_short"}

    # Check if already extracted
    existing = _run_pg(
        "SELECT status FROM extraction_log WHERE qdrant_point_id = %s "
        "AND status = 'completed' LIMIT 1;",
        (memory_id,)
    )
    if existing:
        return {"status": "already_extracted"}

    # Extract triples
    triples = extract_triples(text)

    if not triples:
        _log_extraction(memory_id, memory_id, "completed", 0)
        return {"status": "completed", "triples": 0}

    # Process each triple
    inserted = 0
    low_confidence = 0

    for t in triples:
        if t["confidence"] < CONFIDENCE_THRESHOLD:
            low_confidence += 1
            _log_extraction(memory_id, memory_id, "low_confidence", 0,
                          f"Low confidence triple: {t['subject']} -{t['predicate']}-> {t['object']} ({t['confidence']})")
            continue

        # Upsert entities
        source_id = _upsert_entity(t["subject"], t["subject_type"])
        target_id = _upsert_entity(t["object"], t["object_type"])

        if source_id and target_id:
            _upsert_relationship(source_id, target_id, t["predicate"],
                               {"confidence": t["confidence"], "source_memory": short_id})
            _link_memory_entity(memory_id, source_id)
            _link_memory_entity(memory_id, target_id)
            inserted += 1

    _log_extraction(memory_id, memory_id, "completed", inserted)

    return {
        "status": "completed",
        "triples": inserted,
        "low_confidence": low_confidence,
        "total_extracted": len(triples),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Graph traversal (for retrieval)
# ═══════════════════════════════════════════════════════════════════════════════

def traverse_from_entities(entity_names: list, hops: int = 2, limit: int = 20) -> list:
    """
    Given entity names, traverse the graph 1-2 hops and return connected context.
    Uses recursive CTE in PG.

    Returns list of dicts: {entity, type, predicate, connected_entity, connected_type, depth}
    """
    if not entity_names:
        return []

    # Build WHERE clause for starting entities with params
    name_conditions = " OR ".join("e.name ILIKE %s" for _ in entity_names)
    params = list(entity_names)

    sql = f"""
    WITH RECURSIVE graph_walk AS (
        -- Base: starting entities
        SELECT e.id, e.name, e.type, 0 as depth,
               NULL::text as predicate, NULL::text as from_entity
        FROM entities e
        WHERE {name_conditions}

        UNION ALL

        -- Walk both directions via LATERAL join on a union of inbound+outbound
        SELECT nb.id, nb.name, nb.type, gw.depth + 1,
               nb.predicate, gw.name
        FROM graph_walk gw
        JOIN LATERAL (
            -- Outbound
            SELECT e2.id, e2.name, e2.type, r.predicate
            FROM relationships r
            JOIN entities e2 ON e2.id = r.target_id
            WHERE r.source_id = gw.id
            UNION ALL
            -- Inbound
            SELECT e2.id, e2.name, e2.type, r.predicate
            FROM relationships r
            JOIN entities e2 ON e2.id = r.source_id
            WHERE r.target_id = gw.id
        ) nb ON true
        WHERE gw.depth < %s
    )
    SELECT DISTINCT name, type, depth, predicate, from_entity
    FROM graph_walk
    WHERE depth > 0
    ORDER BY depth, name
    LIMIT %s;
    """
    params.extend([hops, limit])

    result = _run_pg(sql, tuple(params))
    if not result:
        return []

    rows = []
    for line in result.split("\n"):
        if "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) >= 5:
            rows.append({
                "entity": parts[0],
                "type": parts[1],
                "depth": int(parts[2]),
                "predicate": parts[3],
                "from_entity": parts[4],
            })
    return rows


def build_graph_context(entity_names: list) -> str:
    """
    Build a <GRAPH_CONTEXT> block for prompt injection.
    """
    if not entity_names:
        return ""

    connections = traverse_from_entities(entity_names, hops=2, limit=15)
    if not connections:
        return ""

    lines = ["<GRAPH_CONTEXT>"]
    for name in entity_names:
        # Get entity info
        info = _run_pg("SELECT name, type FROM entities WHERE name ILIKE %s LIMIT 1;", (name,))
        if info and "|" in info:
            ename, etype = info.split("|", 1)
            lines.append(f"Entity: {ename} ({etype})")

    # Group connections by depth, dedup
    seen = set()
    for conn in connections:
        key = (conn["predicate"], conn["entity"])
        if key in seen:
            continue
        seen.add(key)
        indent = "  " * conn["depth"]
        pred = conn["predicate"] or "related"
        lines.append(f"{indent}├── {pred} → {conn['entity']} ({conn['type']})")

    lines.append("</GRAPH_CONTEXT>")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Graph entity extraction")
    sub = parser.add_subparsers(dest="command")

    # extract: extract from a single memory
    p_ext = sub.add_parser("extract", help="Extract triples from memory text")
    p_ext.add_argument("text", help="Memory text")
    p_ext.add_argument("--memory-id", default="test-" + str(int(time.time())), help="Memory ID")

    # traverse: walk the graph from entity names
    p_trav = sub.add_parser("traverse", help="Traverse graph from entities")
    p_trav.add_argument("entities", nargs="+", help="Entity names")
    p_trav.add_argument("--hops", type=int, default=2)

    # context: build graph context block
    p_ctx = sub.add_parser("context", help="Build graph context for entities")
    p_ctx.add_argument("entities", nargs="+", help="Entity names")

    # stats: show graph statistics
    sub.add_parser("stats", help="Show graph statistics")

    args = parser.parse_args()

    if args.command == "extract":
        result = extract_and_store(args.text, args.memory_id)
        print(json.dumps(result, indent=2))

    elif args.command == "traverse":
        rows = traverse_from_entities(args.entities, hops=args.hops)
        for r in rows:
            indent = "  " * r["depth"]
            print(f"{indent}{r['from_entity']} --{r['predicate']}--> {r['entity']} ({r['type']})")

    elif args.command == "context":
        ctx = build_graph_context(args.entities)
        print(ctx if ctx else "No graph context found")

    elif args.command == "stats":
        entities = db.pg_query("SELECT COUNT(*) FROM entities;")
        rels = db.pg_query("SELECT COUNT(*) FROM relationships;")
        aliases = db.pg_query("SELECT COUNT(*) FROM entity_aliases;")
        extracted = db.pg_query("SELECT COUNT(*) FROM extraction_log WHERE status = 'completed';")
        print(f"Entities:      {entities}")
        print(f"Relationships: {rels}")
        print(f"Aliases:       {aliases}")
        print(f"Extracted:     {extracted}")

    else:
        parser.print_help()
