#!/usr/bin/env python3
"""
backfill_graph.py — Run graph extraction over existing memories.

One-time script to populate the entity/relationship graph from existing memories.
Processes in batches with rate limiting to avoid OpenRouter throttling.

Usage:
    python3 backfill_graph.py                  # Process all unextracted
    python3 backfill_graph.py --limit 50       # Process 50 memories
    python3 backfill_graph.py --dry-run        # Just count what needs processing
    python3 backfill_graph.py --stats          # Show extraction progress
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

BATCH_SIZE = 10
DELAY_BETWEEN = 2  # seconds between LLM calls (rate limit)


def get_unextracted_memories(limit: int = 100) -> list:
    """Get memories that haven't been through graph extraction yet."""
    rows = db.pg_query(
        "SELECT m.qdrant_point_id, m.summary "
        "FROM memories m "
        "WHERE m.is_deprecated = FALSE "
        "  AND m.summary IS NOT NULL "
        "  AND m.summary != '' "
        "  AND NOT EXISTS ( "
        "      SELECT 1 FROM extraction_log el "
        "      WHERE el.qdrant_point_id = m.qdrant_point_id "
        "        AND el.status = 'completed' "
        "  ) "
        "ORDER BY m.created_at DESC "
        "LIMIT %s;",
        (limit,)
    )

    if not rows:
        return []

    memories = []
    for line in rows.split("\n"):
        if "|" not in line:
            continue
        parts = line.split("|", 1)
        if len(parts) >= 2:
            memories.append({
                "id": parts[0].strip(),
                "summary": parts[1].strip(),
            })
    return memories


def show_stats():
    """Show extraction progress."""
    total = db.pg_query("SELECT COUNT(*) FROM memories WHERE is_deprecated = FALSE;")
    extracted = db.pg_query("SELECT COUNT(DISTINCT qdrant_point_id) FROM extraction_log WHERE status = 'completed';")
    entities = db.pg_query("SELECT COUNT(*) FROM entities;")
    rels = db.pg_query("SELECT COUNT(*) FROM relationships;")
    low_conf = db.pg_query("SELECT COUNT(*) FROM extraction_log WHERE status = 'low_confidence';")

    print(f"Total active memories:  {total}")
    print(f"Extracted:              {extracted}")
    print(f"Remaining:              {int(total or 0) - int(extracted or 0)}")
    print(f"Entities in graph:      {entities}")
    print(f"Relationships:          {rels}")
    print(f"Low-confidence logged:  {low_conf}")


def run_backfill(limit: int = 100, dry_run: bool = False):
    """Process unextracted memories through graph extraction."""
    from graph_extractor import extract_and_store

    memories = get_unextracted_memories(limit)
    if not memories:
        print("No unextracted memories found.")
        return

    print(f"Found {len(memories)} unextracted memories")

    if dry_run:
        for m in memories[:10]:
            print(f"  {m['id'][:8]}  {m['summary'][:70]}")
        if len(memories) > 10:
            print(f"  ... and {len(memories) - 10} more")
        return

    total_triples = 0
    errors = 0
    skipped = 0

    for i, m in enumerate(memories, 1):
        short_id = m["id"][:8]
        summary = m["summary"]

        if len(summary) < 20:
            skipped += 1
            continue

        try:
            result = extract_and_store(summary, m["id"])
            triples = result.get("triples", 0)
            total_triples += triples
            status = result.get("status", "?")

            if triples > 0:
                print(f"  [{i}/{len(memories)}] {short_id}: {triples} triples")
            elif status == "already_extracted":
                skipped += 1
            else:
                pass  # No triples found (normal for simple facts)

        except Exception as e:
            errors += 1
            print(f"  [{i}/{len(memories)}] {short_id}: ERROR {e}")

        # Rate limit
        if i < len(memories):
            time.sleep(DELAY_BETWEEN)

    print(f"\nBackfill complete:")
    print(f"  Processed: {len(memories)}")
    print(f"  Triples extracted: {total_triples}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")


if __name__ == "__main__":
    import argparse

    # Ensure OpenRouter key is available
    if not os.environ.get("OPENROUTER_API_KEY"):
        try:
            with open(os.path.expanduser("~/APIKeys/openrouter.env")) as f:
                for line in f:
                    if line.startswith("OPENROUTER_API_KEY="):
                        os.environ["OPENROUTER_API_KEY"] = line.strip().split("=", 1)[1]
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Backfill graph extraction")
    parser.add_argument("--limit", type=int, default=100, help="Max memories to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    parser.add_argument("--stats", action="store_true", help="Show extraction progress")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    else:
        run_backfill(limit=args.limit, dry_run=args.dry_run)
