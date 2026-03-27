#!/usr/bin/env python3
"""
entity_keyscores_sync.py — Sync entity_boost in keyscores from memory_entities.

Computes a per-memory entity richness score based on:
  1. How many entities are tagged on the memory
  2. How "important" those entities are (inverse document frequency — rare = more valuable)

Score formula:
  entity_boost = min(1.0, sum(idf(entity)) / NORM_FACTOR)

Where idf(entity) = log(total_memories / memories_with_entity)
Normalization keeps most scores in 0.0–0.6 range, with only heavily-tagged
memories hitting 0.8+.

Run via cron every 6 hours (after recency_refresh):
  15 */6 * * * cd /root/ClydeMemory && python3 entity_keyscores_sync.py
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

# Normalization: a memory with 5 average-IDF entities should score ~0.5
NORM_FACTOR = 8.0


def sync():
    """Recompute entity_boost for all memories in keyscores."""

    # Step 1: Get total memory count (non-deprecated)
    total_str = db.pg_query(
        "SELECT COUNT(*) FROM memories WHERE is_deprecated = FALSE;"
    )
    if not total_str:
        print("[entity_sync] No memories found", file=sys.stderr)
        return
    total_memories = int(total_str)
    print(f"[entity_sync] Total active memories: {total_memories}")

    # Step 2: Get entity document frequencies (how many memories each entity appears in)
    df_rows = db.pg_query(
        "SELECT entity_name, COUNT(DISTINCT memory_id) as df "
        "FROM memory_entities me "
        "JOIN memories m ON me.memory_id = m.id "
        "WHERE m.is_deprecated = FALSE "
        "GROUP BY entity_name;"
    )
    if not df_rows:
        print("[entity_sync] No entity data — nothing to sync")
        return

    entity_df = {}
    for line in df_rows.split("\n"):
        if "|" in line:
            parts = line.split("|")
            entity_df[parts[0]] = int(parts[1])

    print(f"[entity_sync] Unique entities: {len(entity_df)}")

    # Step 3: Compute IDF for each entity
    entity_idf = {}
    for name, df in entity_df.items():
        entity_idf[name] = math.log(total_memories / max(df, 1))

    # Step 4: Get all memory → entity mappings
    mem_entities_raw = db.pg_query(
        "SELECT m.id, me.entity_name "
        "FROM memory_entities me "
        "JOIN memories m ON me.memory_id = m.id "
        "WHERE m.is_deprecated = FALSE "
        "ORDER BY m.id;"
    )
    if not mem_entities_raw:
        print("[entity_sync] No memory-entity links found")
        return

    # Group by memory_id
    mem_scores = {}
    for line in mem_entities_raw.split("\n"):
        if "|" not in line:
            continue
        parts = line.split("|")
        mem_id = parts[0]
        ent_name = parts[1]
        if mem_id not in mem_scores:
            mem_scores[mem_id] = 0.0
        mem_scores[mem_id] += entity_idf.get(ent_name, 0.0)

    # Normalize to 0–1 range
    for mem_id in mem_scores:
        mem_scores[mem_id] = min(1.0, mem_scores[mem_id] / NORM_FACTOR)

    print(f"[entity_sync] Memories with entities: {len(mem_scores)}")

    # Step 5: Batch update keyscores
    # Build a single UPDATE using a VALUES list for efficiency
    if not mem_scores:
        return

    # Chunk into batches of 200 to avoid oversized SQL
    items = list(mem_scores.items())
    updated = 0
    batch_size = 200

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        # Use individual updates with params for safety
        for mid, score in batch:
            db.pg_execute(
                "UPDATE keyscores SET entity_boost = %s, computed_at = NOW() "
                "WHERE memory_id = %s::uuid;",
                (round(score, 4), mid)
            )
            updated += 1

    # Step 6: Zero out entity_boost for memories with no entities
    db.pg_execute(
        "UPDATE keyscores SET entity_boost = 0.0, computed_at = NOW() "
        "WHERE memory_id NOT IN ("
        "  SELECT DISTINCT memory_id FROM memory_entities me "
        "  JOIN memories m ON me.memory_id = m.id "
        "  WHERE m.is_deprecated = FALSE"
        ") AND entity_boost > 0;"
    )
    zeroed = 0  # count not available via db.pg_execute

    # Step 7: Recompute composite scores
    db.pg_execute(
        "UPDATE keyscores SET composite_score = compute_composite_score_v2("
        "  recency_score, frequency_score, authority_score, "
        "  entity_boost, impact_score"
        ") WHERE TRUE;"
    )
    composite_count = 0  # count not available via db.pg_execute

    print(f"[entity_sync] Updated entity_boost: {updated} memories")
    if zeroed:
        print(f"[entity_sync] Zeroed out: {zeroed} memories (no entities)")
    print(f"[entity_sync] Recomputed composite: {composite_count} keyscores")


if __name__ == "__main__":
    sync()
