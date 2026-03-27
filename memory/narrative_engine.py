#!/usr/bin/env python3
"""
narrative_engine.py — Narrative Arc Tracking for OpenClaw.

Tracks ongoing "stories" — projects, investigations, learning journeys.
Each narrative has a position in its arc from 0.0 (just started) to 1.0 (complete).

Phase 3 of OpenClaw Cognitive Architecture.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

VALID_ARC_TYPES = {"project", "investigation", "learning", "relationship", "recurring"}
VALID_STATUSES = {"active", "climax", "resolution", "dormant", "completed"}
VALID_EVENT_TYPES = {"start", "milestone", "setback", "breakthrough", "update", "resolution"}

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _log(msg: str):
    print(f"[narrative_engine] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════════════════════

def create_narrative(title: str, description: str = "", arc_type: str = "project",
                     importance: float = 0.5, tags: list = None) -> int:
    """Create a new narrative arc. Returns narrative ID or -1 on failure."""
    if arc_type not in VALID_ARC_TYPES:
        _log(f"Invalid arc_type '{arc_type}' — using 'project'")
        arc_type = "project"

    tags = tags or []

    result = db.pg_query(
        "INSERT INTO narratives (title, description, arc_type, importance, tags) "
        "VALUES (%s, %s, %s, %s, %s::text[]) "
        "ON CONFLICT (title) DO UPDATE SET "
        "  description = EXCLUDED.description, "
        "  arc_type = EXCLUDED.arc_type, "
        "  importance = EXCLUDED.importance, "
        "  tags = EXCLUDED.tags, "
        "  updated_at = NOW() "
        "RETURNING id;",
        (title, description, arc_type, importance, tags)
    )
    if result:
        nid = int(result.strip())
        _log(f"Created/updated narrative '{title}' (id={nid})")
        return nid
    _log(f"Failed to create narrative '{title}'")
    return -1


def add_event(narrative_id_or_title, event_type: str, description: str,
              position_delta: float = 0.0):
    """
    Add an event to a narrative and update its position.

    Args:
        narrative_id_or_title: int ID or string title
        event_type: start, milestone, setback, breakthrough, update, resolution
        description: what happened
        position_delta: how much to move the arc (-1.0 to 1.0)
    """
    if event_type not in VALID_EVENT_TYPES:
        _log(f"Invalid event_type '{event_type}' — using 'update'")
        event_type = "update"

    # Resolve narrative ID
    if isinstance(narrative_id_or_title, int):
        nid = narrative_id_or_title
    else:
        result = db.pg_query("SELECT id FROM narratives WHERE title = %s", (str(narrative_id_or_title),))
        if not result:
            _log(f"Narrative '{narrative_id_or_title}' not found")
            return
        nid = int(result.strip())

    # Insert event
    db.pg_execute(
        "INSERT INTO narrative_events (narrative_id, event_type, description, position_delta) "
        "VALUES (%s, %s, %s, %s);",
        (nid, event_type, description, position_delta)
    )

    # Update narrative position (clamped 0.0-1.0) and last_referenced
    db.pg_execute(
        "UPDATE narratives "
        "SET position = GREATEST(0.0, LEAST(1.0, position + %s)), "
        "    last_referenced = NOW(), "
        "    updated_at = NOW() "
        "WHERE id = %s;",
        (position_delta, nid)
    )

    # Auto-update status based on event type
    if event_type == "resolution":
        db.pg_execute("UPDATE narratives SET status = 'completed', position = 1.0 WHERE id = %s;", (nid,))
    elif event_type == "breakthrough":
        # Check if position is high enough for climax
        pos = db.pg_query("SELECT position FROM narratives WHERE id = %s", (nid,))
        if pos and float(pos.strip()) >= 0.7:
            db.pg_execute("UPDATE narratives SET status = 'climax' WHERE id = %s;", (nid,))

    _log(f"Added {event_type} event to narrative {nid}: {description}")


def get_active_narratives(limit: int = 10) -> list:
    """Get active/climax narratives ordered by importance and recency."""
    rows = db.pg_query(
        "SELECT n.id, n.title, n.description, n.arc_type, n.status, "
        "       n.position, n.importance, n.created_at, n.updated_at, n.tags "
        "FROM narratives n "
        "WHERE n.status IN ('active', 'climax') "
        "ORDER BY n.importance DESC, n.updated_at DESC "
        "LIMIT %s",
        (limit,)
    )
    if not rows:
        return []

    result = []
    for row in rows.split("\n"):
        row = row.strip()
        if not row:
            continue
        parts = row.split("|")
        if len(parts) < 10:
            continue
        result.append({
            "id": int(parts[0]),
            "title": parts[1],
            "description": parts[2],
            "arc_type": parts[3],
            "status": parts[4],
            "position": float(parts[5]),
            "importance": float(parts[6]),
            "created_at": parts[7],
            "updated_at": parts[8],
            "tags": parts[9],
        })
    return result


def get_narrative_position(query: str) -> float:
    """
    Get weighted average position of narratives matching a query.

    Uses word-level matching against titles, descriptions, and entity links.
    Any query word (3+ chars) matching via ILIKE counts as a hit.
    Returns 0.0-1.0 indicating aggregate narrative position.
    """
    import re as _re
    words = [w for w in _re.findall(r'[a-zA-Z]{3,}', query) if w.lower() not in {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "was", "one", "our", "out", "has", "how", "its", "may", "new", "now",
        "this", "that", "with", "have", "from", "they", "been", "some", "what",
        "when", "will", "more", "into", "also", "than", "them", "very", "just",
    }]
    if not words:
        return 0.0

    # Build OR clause: any word matches title, description, or entity name
    word_clauses = []
    params = []
    for w in words:
        wp = '%' + w + '%'
        word_clauses.append(
            "(n.title ILIKE %s OR n.description ILIKE %s OR nel.entity_name ILIKE %s)"
        )
        params.extend([wp, wp, wp])
    where_words = " OR ".join(word_clauses)

    rows = db.pg_query(
        "SELECT DISTINCT n.id, n.position, n.importance "
        "FROM narratives n "
        "LEFT JOIN narrative_entity_links nel ON nel.narrative_id = n.id "
        "WHERE n.status IN ('active', 'climax') "
        f"AND ({where_words})",
        tuple(params)
    )
    if not rows:
        return 0.0

    total_weight = 0.0
    weighted_pos = 0.0
    for row in rows.split("\n"):
        row = row.strip()
        if not row:
            continue
        parts = row.split("|")
        if len(parts) < 3:
            continue
        pos = float(parts[1])
        imp = float(parts[2])
        weighted_pos += pos * imp
        total_weight += imp

    if total_weight == 0:
        return 0.0
    return weighted_pos / total_weight


def get_narrative_context(entity_names: list = None) -> str:
    """Format active narratives for prompt injection, optionally filtered by entities."""
    if entity_names:
        # Get narratives linked to these entities
        entity_filter = " OR ".join("nel.entity_name ILIKE %s" for _ in entity_names)
        entity_params = ['%' + e + '%' for e in entity_names]
        rows = db.pg_query(
            "SELECT DISTINCT n.id, n.title, n.arc_type, n.status, n.position, n.importance "
            "FROM narratives n "
            "LEFT JOIN narrative_entity_links nel ON nel.narrative_id = n.id "
            "WHERE n.status IN ('active', 'climax') "
            f"AND ({entity_filter}) "
            "ORDER BY n.importance DESC "
            "LIMIT 10",
            tuple(entity_params)
        )
    else:
        rows = db.pg_query(
            "SELECT id, title, arc_type, status, position, importance "
            "FROM narratives "
            "WHERE status IN ('active', 'climax') "
            "ORDER BY importance DESC "
            "LIMIT 10"
        )

    if not rows:
        return ""

    lines = ["## Active Narratives"]
    for row in rows.split("\n"):
        row = row.strip()
        if not row:
            continue
        parts = row.split("|")
        if len(parts) < 6:
            continue
        nid, title, arc_type, status, position, importance = parts[:6]
        pos = float(position)
        imp = float(importance)
        arc_bar = _arc_bar(pos)

        # Get recent events for this narrative
        recent = db.pg_query(
            "SELECT event_type, description "
            "FROM narrative_events "
            "WHERE narrative_id = %s "
            "ORDER BY created_at DESC "
            "LIMIT 1",
            (int(nid),)
        )
        last_event = ""
        if recent:
            eparts = recent.strip().split("|", 1)
            if len(eparts) == 2:
                last_event = f" (last: {eparts[0]} - {eparts[1]})"

        status_icon = {"active": "->", "climax": "!!"}
        icon = status_icon.get(status, "->")
        lines.append(f"- {icon} **{title}** [{arc_type}] {arc_bar} importance={imp:.1f}{last_event}")

    return "\n".join(lines)


def _arc_bar(position: float) -> str:
    """Visual arc position: [====>-----------] 0.3"""
    total = 20
    pos = int(position * total)
    pos = max(0, min(total, pos))
    return "[" + "=" * pos + ">" + "-" * (total - pos) + f"] {position:.1f}"


def update_narrative_status(narrative_id_or_title, status: str):
    """Update a narrative's status."""
    if status not in VALID_STATUSES:
        _log(f"Invalid status '{status}'")
        return

    if isinstance(narrative_id_or_title, int):
        db.pg_execute("UPDATE narratives SET status = %s, updated_at = NOW() WHERE id = %s;",
                      (status, narrative_id_or_title))
    else:
        db.pg_execute("UPDATE narratives SET status = %s, updated_at = NOW() WHERE title = %s;",
                      (status, str(narrative_id_or_title)))
    _log(f"Updated narrative status to '{status}'")


def link_entity(narrative_id_or_title, entity_name: str, role: str = "involved"):
    """Link an entity to a narrative."""
    if isinstance(narrative_id_or_title, int):
        nid = narrative_id_or_title
    else:
        result = db.pg_query("SELECT id FROM narratives WHERE title = %s", (str(narrative_id_or_title),))
        if not result:
            _log(f"Narrative '{narrative_id_or_title}' not found")
            return
        nid = int(result.strip())

    db.pg_execute(
        "INSERT INTO narrative_entity_links (narrative_id, entity_name, role) "
        "VALUES (%s, %s, %s) "
        "ON CONFLICT (narrative_id, entity_name) DO UPDATE SET role = EXCLUDED.role;",
        (nid, entity_name, role)
    )
    _log(f"Linked entity '{entity_name}' to narrative {nid} as '{role}'")


def seed_narratives():
    """Seed initial narratives from known projects."""
    narratives = [
        ("OpenClaw Cognitive Architecture",
         "Building persistent memory, knowledge graph, goal tracking, tension tracking, and narrative arcs for Luther/OpenClaw.",
         "project", 0.9,
         ["openclaw", "memory", "cognitive"],
         [("OpenClaw", "protagonist"), ("Mem0", "tool"), ("Qdrant", "tool"),
          ("PostgreSQL", "tool"), ("Redis", "tool")]),
        ("ANJ-12KP Solar System",
         "Solar panel installation and monitoring project for ANJ-12KP system.",
         "project", 0.6,
         ["solar", "hardware", "monitoring"],
         [("ANJ-12KP", "protagonist")]),
        ("Shopping Worker HTPC",
         "Home theater PC build or configuration project for shopping/media worker.",
         "project", 0.5,
         ["htpc", "hardware", "media"],
         []),
    ]

    for title, desc, arc_type, importance, tags, entities in narratives:
        nid = create_narrative(title, desc, arc_type, importance, tags)
        if nid > 0:
            for entity_name, role in entities:
                link_entity(nid, entity_name, role)

    # Add a start event for OpenClaw
    add_event("OpenClaw Cognitive Architecture", "milestone",
              "Phase 3: tension tracking and narrative engine built", 0.15)

    _log("Seeded initial narratives")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print('  narrative_engine.py create "title" [--type TYPE] [--importance N] [--desc "..."] [--tags t1,t2]')
        print('  narrative_engine.py event "title" <event_type> "description" [position_delta]')
        print("  narrative_engine.py list")
        print('  narrative_engine.py position "query"')
        print("  narrative_engine.py context [entity1,entity2,...]")
        print('  narrative_engine.py status "title" <new_status>')
        print('  narrative_engine.py link "title" "entity_name" [role]')
        print("  narrative_engine.py seed")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "seed":
        seed_narratives()

    elif cmd == "create":
        if len(sys.argv) < 3:
            print("Usage: narrative_engine.py create \"title\" [--type TYPE] [--importance N]")
            sys.exit(1)
        title = sys.argv[2]
        # Parse optional flags
        arc_type = "project"
        importance = 0.5
        desc = ""
        tags = []
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--type" and i + 1 < len(sys.argv):
                arc_type = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--importance" and i + 1 < len(sys.argv):
                importance = float(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--desc" and i + 1 < len(sys.argv):
                desc = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--tags" and i + 1 < len(sys.argv):
                tags = sys.argv[i + 1].split(",")
                i += 2
            else:
                i += 1
        nid = create_narrative(title, desc, arc_type, importance, tags)
        print(f"Created narrative id={nid}")

    elif cmd == "event":
        if len(sys.argv) < 5:
            print('Usage: narrative_engine.py event "title" <event_type> "description" [position_delta]')
            sys.exit(1)
        title = sys.argv[2]
        event_type = sys.argv[3]
        description = sys.argv[4]
        delta = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
        add_event(title, event_type, description, delta)

    elif cmd == "list":
        narratives = get_active_narratives()
        if not narratives:
            print("No active narratives. Run 'seed' first.")
        for n in narratives:
            bar = _arc_bar(n["position"])
            print(f"[{n['id']}] {n['title']} ({n['arc_type']}/{n['status']}) {bar} imp={n['importance']}")

    elif cmd == "position":
        if len(sys.argv) < 3:
            print('Usage: narrative_engine.py position "query"')
            sys.exit(1)
        pos = get_narrative_position(sys.argv[2])
        print(f"Weighted position: {pos:.3f}")

    elif cmd == "context":
        entities = None
        if len(sys.argv) > 2:
            entities = sys.argv[2].split(",")
        print(get_narrative_context(entities))

    elif cmd == "status":
        if len(sys.argv) < 4:
            print('Usage: narrative_engine.py status "title" <new_status>')
            sys.exit(1)
        update_narrative_status(sys.argv[2], sys.argv[3])

    elif cmd == "link":
        if len(sys.argv) < 4:
            print('Usage: narrative_engine.py link "title" "entity_name" [role]')
            sys.exit(1)
        role = sys.argv[4] if len(sys.argv) > 4 else "involved"
        link_entity(sys.argv[2], sys.argv[3], role)

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
