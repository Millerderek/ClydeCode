#!/usr/bin/env python3
"""
recency_refresh.py — Recompute recency_score for all memories.

Pinned memories always get recency_score = 1.0.
All others use exponential decay: exp(-0.023 * age_days) → 30-day halflife.

Run via cron every 6 hours:
  0 */6 * * * cd /root/ClydeMemory && python3 recency_refresh.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

DECAY_RATE = 0.023  # ln(2)/30 ≈ 0.023 → 30-day halflife


def refresh():
    conn = db.get_pg()
    cur = conn.cursor()

    sql_recency = (
        "UPDATE keyscores k "
        "SET recency_score = CASE "
        "    WHEN m.pinned = TRUE THEN 1.0 "
        "    ELSE compute_recency_score(m.last_accessed, %s) "
        "END, "
        "computed_at = NOW() "
        "FROM memories m "
        "WHERE k.memory_id = m.id "
        "  AND m.is_deprecated = FALSE;"
    )

    sql_composite = (
        "UPDATE keyscores "
        "SET composite_score = compute_composite_score_v2( "
        "    recency_score, frequency_score, authority_score, "
        "    entity_boost, impact_score "
        ") "
        "WHERE TRUE;"
    )

    cur.execute(sql_recency, (DECAY_RATE,))
    print(f"[recency_refresh] Updated {cur.rowcount} keyscores")
    cur.execute(sql_composite)
    print(f"[recency_refresh] Recomputed {cur.rowcount} composite scores")


if __name__ == "__main__":
    refresh()
