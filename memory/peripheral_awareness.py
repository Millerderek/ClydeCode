#!/usr/bin/env python3
"""Phase 4 — Peripheral Awareness: Background Pattern Detection for OpenClaw.

Runs periodically (cron, every 2 hours) to detect patterns, anomalies, and
proactive context that the agent should be aware of but was not explicitly
asked about.
"""

import json
import sys
from datetime import datetime, timezone
from typing import Any

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

# ── helpers ────────────────────────────────────────────────────────────────


def _pg(sql: str, params=None) -> list[str]:
    """Run a SQL statement via db module and return cleaned output lines."""
    result = db.pg_query(sql, params)
    if not result:
        return []
    return [l for l in result.strip().splitlines() if l]


def _pg_one(sql: str, params=None) -> str | None:
    rows = _pg(sql, params)
    return rows[0] if rows else None


# ── signal storage ─────────────────────────────────────────────────────────

def store_signal(
    signal_type: str,
    title: str,
    description: str,
    confidence: float = 0.7,
    priority: str = "low",
    related_entities: list[str] | None = None,
    source_query: str | None = None,
    expires_hours: int = 72,
) -> int | None:
    """Insert an awareness signal and return its id."""
    row = _pg_one(
        "INSERT INTO awareness_signals "
        "(signal_type, title, description, confidence, priority, "
        "related_entities, source_query, expires_at) "
        "VALUES (%s, %s, %s, %s, %s, %s::text[], %s, "
        "NOW() + interval '%s hours') "
        "RETURNING id",
        (signal_type, title, description, confidence, priority,
         related_entities, source_query, expires_hours)
    )
    return int(row) if row else None


# ── signal management ──────────────────────────────────────────────────────

def get_pending_signals(limit: int = 10) -> list[dict]:
    """Return top pending signals ordered by priority then recency."""
    sql = (
        "SELECT id, signal_type, title, description, confidence, priority, "
        "related_entities, created_at "
        "FROM awareness_signals "
        "WHERE status = 'pending' "
        "AND (expires_at IS NULL OR expires_at > NOW()) "
        "ORDER BY "
        "  CASE priority WHEN 'high' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END, "
        "  created_at DESC "
        f"LIMIT {limit}"
    )
    rows = _pg(sql)
    signals = []
    for row in rows:
        if not row:
            continue
        parts = row.split("|")
        if len(parts) < 8:
            continue
        signals.append({
            "id": int(parts[0]),
            "signal_type": parts[1],
            "title": parts[2],
            "description": parts[3],
            "confidence": float(parts[4]),
            "priority": parts[5],
            "related_entities": parts[6],
            "created_at": parts[7],
        })
    return signals


def surface_signal(signal_id: int) -> None:
    """Mark a signal as surfaced."""
    db.pg_execute(
        "UPDATE awareness_signals SET status='surfaced', surfaced_at=NOW() WHERE id = %s",
        (signal_id,),
    )


def dismiss_signal(signal_id: int) -> None:
    """Mark a signal as dismissed."""
    db.pg_execute(
        "UPDATE awareness_signals SET status='dismissed' WHERE id = %s",
        (signal_id,),
    )


# ── detectors ──────────────────────────────────────────────────────────────

def detect_cooccurrence_spikes(days_recent: int = 7, threshold: float = 3.0) -> list[dict]:
    """Find entity pairs co-occurring much more in recent days vs historical avg.

    Strategy: count co-occurrence in last `days_recent` days and compare to
    the per-week average over the full history.  Flag pairs where the recent
    rate exceeds `threshold` x the historical weekly rate.
    """
    sql = (
        "WITH recent_pairs AS ( "
        "  SELECT a.entity_name AS e1, b.entity_name AS e2, COUNT(*) AS cnt "
        "  FROM memory_entities a "
        "  JOIN memory_entities b ON a.memory_id = b.memory_id AND a.entity_name < b.entity_name "
        "  JOIN memories m ON m.id = a.memory_id "
        f"  WHERE m.created_at >= NOW() - interval '{days_recent} days' "
        "  GROUP BY a.entity_name, b.entity_name "
        "  HAVING COUNT(*) >= 2 "
        "), "
        "historical_pairs AS ( "
        "  SELECT a.entity_name AS e1, b.entity_name AS e2, COUNT(*) AS cnt, "
        "    GREATEST(1, EXTRACT(EPOCH FROM (NOW() - MIN(m.created_at))) / 604800) AS weeks "
        "  FROM memory_entities a "
        "  JOIN memory_entities b ON a.memory_id = b.memory_id AND a.entity_name < b.entity_name "
        "  JOIN memories m ON m.id = a.memory_id "
        f"  WHERE m.created_at < NOW() - interval '{days_recent} days' "
        "  GROUP BY a.entity_name, b.entity_name "
        ") "
        "SELECT r.e1, r.e2, r.cnt AS recent_cnt, "
        "  COALESCE(h.cnt / h.weeks, 0) AS hist_weekly "
        "FROM recent_pairs r "
        "LEFT JOIN historical_pairs h ON r.e1 = h.e1 AND r.e2 = h.e2 "
        f"WHERE r.cnt > COALESCE(h.cnt / h.weeks, 0) * {threshold} "
        "ORDER BY r.cnt DESC "
        "LIMIT 20"
    )
    rows = _pg(sql)
    signals = []
    for row in rows:
        if not row:
            continue
        parts = row.split("|")
        if len(parts) < 4:
            continue
        e1, e2, recent, hist = parts[0], parts[1], parts[2], parts[3]
        sig = {
            "signal_type": "pattern",
            "title": f"Emerging connection: {e1} <-> {e2}",
            "description": (
                f"Co-occurrence spiked to {recent} times in last {days_recent} days "
                f"(historical weekly avg: {hist})."
            ),
            "confidence": min(0.9, 0.5 + float(recent) * 0.05),
            "priority": "normal",
            "related_entities": [e1, e2],
            "source_query": "detect_cooccurrence_spikes",
        }
        signals.append(sig)
    return signals


def detect_stale_critical(days: int = 30) -> list[dict]:
    """Find critical/high-impact memories not accessed in `days`+ days."""
    sql = (
        "SELECT id, summary, impact_category, last_accessed, created_at "
        "FROM memories "
        "WHERE impact_category IN ('critical', 'high') "
        "  AND is_deprecated = false "
        f"  AND COALESCE(last_accessed, created_at) < NOW() - interval '{days} days' "
        "ORDER BY impact_category, COALESCE(last_accessed, created_at) "
        "LIMIT 20"
    )
    rows = _pg(sql)
    signals = []
    for row in rows:
        if not row:
            continue
        parts = row.split("|")
        if len(parts) < 5:
            continue
        mid, summary, impact, last_acc, created = (
            parts[0], parts[1], parts[2], parts[3], parts[4],
        )
        summary_short = (summary or "no summary")[:120]
        sig = {
            "signal_type": "stale_warning",
            "title": f"Stale {impact} memory: {summary_short}",
            "description": (
                f"Memory {mid[:8]}... ({impact}) last accessed {last_acc or created}. "
                f"Over {days} days without review."
            ),
            "confidence": 0.8,
            "priority": "high" if impact == "critical" else "normal",
            "related_entities": [],
            "source_query": "detect_stale_critical",
        }
        signals.append(sig)
    return signals


def detect_unresolved_contradictions(days: int = 7) -> list[dict]:
    """Find contradictions unresolved for more than `days` days."""
    sql = (
        "SELECT c.id, c.memory_a_id, c.memory_b_id, c.detected_at, "
        "  ma.summary AS sum_a, mb.summary AS sum_b "
        "FROM contradictions c "
        "LEFT JOIN memories ma ON ma.id = c.memory_a_id "
        "LEFT JOIN memories mb ON mb.id = c.memory_b_id "
        "WHERE c.resolved = false "
        f"  AND c.detected_at < NOW() - interval '{days} days' "
        "ORDER BY c.detected_at "
        "LIMIT 20"
    )
    rows = _pg(sql)
    signals = []
    for row in rows:
        if not row:
            continue
        parts = row.split("|")
        if len(parts) < 6:
            continue
        cid, ma_id, mb_id, detected, sum_a, sum_b = (
            parts[0], parts[1], parts[2], parts[3], parts[4], parts[5],
        )
        sa = (sum_a or ma_id[:8])[:80]
        sb = (sum_b or mb_id[:8])[:80]
        sig = {
            "signal_type": "anomaly",
            "title": f"Unresolved contradiction #{cid}",
            "description": (
                f"Contradiction between [{sa}] and [{sb}] "
                f"detected {detected}, still unresolved after {days}+ days."
            ),
            "confidence": 0.85,
            "priority": "normal",
            "related_entities": [],
            "source_query": "detect_unresolved_contradictions",
        }
        signals.append(sig)
    return signals


def detect_orphan_entities(min_connections: int = 2) -> list[dict]:
    """Find entities with fewer than `min_connections` relationships."""
    sql = (
        "SELECT e.id, e.name, e.type, "
        "  (SELECT COUNT(*) FROM relationships r "
        "   WHERE r.source_id = e.id OR r.target_id = e.id) AS rel_count "
        "FROM entities e "
        "WHERE (SELECT COUNT(*) FROM relationships r "
        "       WHERE r.source_id = e.id OR r.target_id = e.id) "
        f"  < {min_connections} "
        "ORDER BY rel_count, e.name "
        "LIMIT 30"
    )
    rows = _pg(sql)
    signals = []
    for row in rows:
        if not row:
            continue
        parts = row.split("|")
        if len(parts) < 4:
            continue
        eid, name, etype, rel_count = parts[0], parts[1], parts[2], parts[3]
        sig = {
            "signal_type": "connection",
            "title": f"Orphan entity: {name} ({etype})",
            "description": (
                f"Entity '{name}' has only {rel_count} relationship(s). "
                f"May need context enrichment or linking."
            ),
            "confidence": 0.6,
            "priority": "low",
            "related_entities": [name],
            "source_query": "detect_orphan_entities",
        }
        signals.append(sig)
    return signals


def detect_topic_drift() -> list[dict]:
    """Detect topics whose recent entity neighbours differ from early ones.

    Simplified approach: for each entity that appears in at least 6 memories,
    compare the set of co-occurring entities in the oldest third of its
    memories vs the newest third.  If overlap (Jaccard) is low, flag drift.
    """
    sql = (
        "SELECT entity_name, COUNT(DISTINCT memory_id) AS mcnt "
        "FROM memory_entities "
        "GROUP BY entity_name "
        "HAVING COUNT(DISTINCT memory_id) >= 6 "
        "ORDER BY mcnt DESC "
        "LIMIT 50"
    )
    candidates = _pg(sql)
    signals = []

    for cand in candidates:
        if not cand:
            continue
        parts = cand.split("|")
        if len(parts) < 2:
            continue
        entity_name = parts[0]

        old_neighbours_sql = (
            "WITH ranked AS ( "
            "  SELECT a.memory_id, m.created_at, "
            "    NTILE(3) OVER (ORDER BY m.created_at) AS tercile "
            "  FROM memory_entities a "
            "  JOIN memories m ON m.id = a.memory_id "
            "  WHERE a.entity_name = %s "
            ") "
            "SELECT DISTINCT b.entity_name "
            "FROM ranked r "
            "JOIN memory_entities b ON r.memory_id = b.memory_id "
            "  AND b.entity_name != %s "
            "WHERE r.tercile = 1"
        )
        new_neighbours_sql = (
            "WITH ranked AS ( "
            "  SELECT a.memory_id, m.created_at, "
            "    NTILE(3) OVER (ORDER BY m.created_at) AS tercile "
            "  FROM memory_entities a "
            "  JOIN memories m ON m.id = a.memory_id "
            "  WHERE a.entity_name = %s "
            ") "
            "SELECT DISTINCT b.entity_name "
            "FROM ranked r "
            "JOIN memory_entities b ON r.memory_id = b.memory_id "
            "  AND b.entity_name != %s "
            "WHERE r.tercile = 3"
        )

        old_set = set(r for r in _pg(old_neighbours_sql, (entity_name, entity_name)) if r)
        new_set = set(r for r in _pg(new_neighbours_sql, (entity_name, entity_name)) if r)

        if not old_set and not new_set:
            continue

        union = old_set | new_set
        intersection = old_set & new_set
        jaccard = len(intersection) / len(union) if union else 1.0

        if jaccard < 0.35 and len(union) >= 3:
            dropped = old_set - new_set
            gained = new_set - old_set
            dropped_str = ", ".join(sorted(dropped)[:5]) or "none"
            gained_str = ", ".join(sorted(gained)[:5]) or "none"
            sig = {
                "signal_type": "pattern",
                "title": f"Topic drift: {entity_name}",
                "description": (
                    f"Entity '{entity_name}' context shifted (Jaccard={jaccard:.2f}). "
                    f"Lost neighbours: [{dropped_str}]. "
                    f"Gained neighbours: [{gained_str}]."
                ),
                "confidence": round(min(0.9, 0.5 + (1 - jaccard) * 0.5), 2),
                "priority": "normal",
                "related_entities": [entity_name],
                "source_query": "detect_topic_drift",
            }
            signals.append(sig)

    return signals


# ── orchestration ──────────────────────────────────────────────────────────

def run_scan() -> dict:
    """Run all detectors, store signals, return summary."""
    # Expire old pending signals first
    _pg(
        "UPDATE awareness_signals SET status='dismissed' "
        "WHERE status='pending' AND expires_at IS NOT NULL AND expires_at < NOW()"
    )

    detectors = [
        ("cooccurrence_spikes", detect_cooccurrence_spikes),
        ("stale_critical", detect_stale_critical),
        ("unresolved_contradictions", detect_unresolved_contradictions),
        ("orphan_entities", detect_orphan_entities),
        ("topic_drift", detect_topic_drift),
    ]

    summary: dict[str, Any] = {}
    total_stored = 0

    for name, fn in detectors:
        try:
            signals = fn()
            stored = 0
            for sig in signals:
                sid = store_signal(
                    signal_type=sig["signal_type"],
                    title=sig["title"],
                    description=sig["description"],
                    confidence=sig.get("confidence", 0.7),
                    priority=sig.get("priority", "low"),
                    related_entities=sig.get("related_entities"),
                    source_query=sig.get("source_query"),
                )
                if sid is not None:
                    stored += 1
            summary[name] = {"detected": len(signals), "stored": stored}
            total_stored += stored
        except Exception as exc:
            summary[name] = {"error": str(exc)}

    summary["total_stored"] = total_stored
    return summary


# ── prompt injection builder ───────────────────────────────────────────────

def build_awareness_context(limit: int = 3) -> str:
    """Build a <PERIPHERAL_AWARENESS> block for prompt injection."""
    signals = get_pending_signals(limit=limit)
    if not signals:
        return ""

    lines = ["<PERIPHERAL_AWARENESS>"]
    for sig in signals:
        prio = sig["priority"].upper()
        lines.append(
            f"[{prio}] ({sig['signal_type']}) {sig['title']}\n"
            f"  {sig['description']}\n"
            f"  confidence={sig['confidence']}  id={sig['id']}"
        )
    lines.append("</PERIPHERAL_AWARENESS>")
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────

def _print_signals(signals: list[dict]) -> None:
    for s in signals:
        prio = s["priority"].upper()
        print(f"  #{s['id']:>4}  [{prio:>6}] ({s['signal_type']}) {s['title']}")
        print(f"         {s['description'][:120]}")
        print(f"         confidence={s['confidence']}  created={s['created_at']}")
        print()


def cli():
    if len(sys.argv) < 2:
        print("Usage: peripheral_awareness.py <scan|list|surface ID|dismiss ID|context>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "scan":
        print("Running peripheral awareness scan...")
        summary = run_scan()
        print()
        for name, info in summary.items():
            if name == "total_stored":
                continue
            if "error" in info:
                print(f"  {name}: ERROR -- {info['error']}")
            else:
                print(f"  {name}: detected={info['detected']}, stored={info['stored']}")
        print(f"\n  Total new signals stored: {summary.get('total_stored', 0)}")
        print()
        pending = get_pending_signals(limit=5)
        if pending:
            print("Top pending signals:")
            _print_signals(pending)
        else:
            print("No pending signals.")

    elif cmd == "list":
        signals = get_pending_signals(limit=20)
        if signals:
            print(f"Pending signals ({len(signals)}):\n")
            _print_signals(signals)
        else:
            print("No pending signals.")

    elif cmd == "surface":
        if len(sys.argv) < 3:
            print("Usage: peripheral_awareness.py surface <id>")
            sys.exit(1)
        sid = int(sys.argv[2])
        surface_signal(sid)
        print(f"Signal #{sid} marked as surfaced.")

    elif cmd == "dismiss":
        if len(sys.argv) < 3:
            print("Usage: peripheral_awareness.py dismiss <id>")
            sys.exit(1)
        sid = int(sys.argv[2])
        dismiss_signal(sid)
        print(f"Signal #{sid} dismissed.")

    elif cmd == "context":
        ctx = build_awareness_context(limit=3)
        if ctx:
            print(ctx)
        else:
            print("(no pending signals to surface)")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
