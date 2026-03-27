#!/usr/bin/env python3
"""
tension_tracker.py — Developmental Tension Tracking for OpenClaw.

Tracks unresolved tensions from soul.md without forcing false resolution.
Uses exponential moving average to maintain a lean_score for each tension.

Phase 3 of OpenClaw Cognitive Architecture.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

EMA_ALPHA = 0.15  # Exponential moving average smoothing factor

# The 5 core developmental tensions from soul.md
CORE_TENSIONS = [
    ("speed_vs_thoroughness", "speed", "thoroughness",
     "Tension between moving fast and being thorough. Speed risks missing details; thoroughness risks stalling."),
    ("autonomy_vs_oversight", "autonomy", "oversight",
     "Tension between acting independently and seeking user confirmation. Autonomy risks mistakes; oversight risks slowness."),
    ("creativity_vs_precision", "creativity", "precision",
     "Tension between creative exploration and precise execution. Creativity risks imprecision; precision risks rigidity."),
    ("warmth_vs_bluntness", "warmth", "bluntness",
     "Tension between empathetic warmth and direct bluntness. Warmth risks vagueness; bluntness risks coldness."),
    ("loyalty_vs_judgment", "loyalty to user intent", "independent judgment",
     "Tension between following user intent exactly and exercising independent judgment. Loyalty risks blind obedience; judgment risks overreach."),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _log(msg: str):
    print(f"[tension_tracker] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════════════════════

def seed_tensions():
    """Insert the 5 core developmental tensions if they don't already exist."""
    for name, pole_a, pole_b, desc in CORE_TENSIONS:
        db.pg_execute(
            "INSERT INTO tensions (name, pole_a, pole_b, description) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (name) DO NOTHING;",
            (name, pole_a, pole_b, desc)
        )
        db.pg_execute(
            "INSERT INTO tension_states (tension_id, lean_score, observation_count) "
            "SELECT id, 0.5, 0 FROM tensions WHERE name = %s "
            "ON CONFLICT (tension_id) DO NOTHING;",
            (name,)
        )
    _log("Seeded 5 core tensions")


def observe_tension(tension_name: str, resolution: str, context: str = "", confidence: float = 0.7):
    """
    Record a tension observation and update lean_score via EMA.

    Args:
        tension_name: e.g. "speed_vs_thoroughness"
        resolution: 'pole_a', 'pole_b', or 'balanced'
        context: what was happening when this was observed
        confidence: 0.0-1.0 how confident in this observation
    """
    if resolution not in ("pole_a", "pole_b", "balanced"):
        _log(f"Invalid resolution '{resolution}' — must be pole_a, pole_b, or balanced")
        return

    # Get tension ID
    tid = db.pg_query("SELECT id FROM tensions WHERE name = %s", (tension_name,))
    if not tid:
        _log(f"Tension '{tension_name}' not found")
        return

    tid = tid.strip()

    # Insert observation
    db.pg_execute(
        "INSERT INTO tension_observations (tension_id, resolution, context, confidence) "
        "VALUES (%s, %s, %s, %s);",
        (int(tid), resolution, context, confidence)
    )

    # Calculate observation value: 0.0 = pole_a, 0.5 = balanced, 1.0 = pole_b
    if resolution == "pole_a":
        obs_value = 0.0
    elif resolution == "pole_b":
        obs_value = 1.0
    else:
        obs_value = 0.5

    # Update lean_score using EMA: new = alpha * observation + (1 - alpha) * old
    alpha = EMA_ALPHA * confidence  # Scale alpha by confidence
    db.pg_execute(
        "UPDATE tension_states "
        "SET lean_score = %s * %s + (1.0 - %s) * lean_score, "
        "    observation_count = observation_count + 1, "
        "    last_observed = NOW(), "
        "    updated_at = NOW() "
        "WHERE tension_id = %s;",
        (alpha, obs_value, alpha, int(tid))
    )

    _log(f"Observed {tension_name}: {resolution} (confidence={confidence})")


def get_tension_state(tension_name: str) -> dict:
    """Get current state of a tension."""
    row = db.pg_query(
        "SELECT t.name, t.pole_a, t.pole_b, t.description, "
        "       ts.lean_score, ts.observation_count, ts.last_observed "
        "FROM tensions t "
        "JOIN tension_states ts ON ts.tension_id = t.id "
        "WHERE t.name = %s",
        (tension_name,)
    )
    if not row:
        return {}

    parts = row.split("|")
    if len(parts) < 7:
        return {}

    lean = float(parts[4])
    return {
        "name": parts[0],
        "pole_a": parts[1],
        "pole_b": parts[2],
        "description": parts[3],
        "lean_score": lean,
        "observation_count": int(parts[5]),
        "last_observed": parts[6] or None,
        "interpretation": _interpret_lean(parts[1], parts[2], lean),
    }


def get_all_tensions() -> list:
    """Get all tension states with interpretations."""
    rows = db.pg_query(
        "SELECT t.name, t.pole_a, t.pole_b, t.description, "
        "       ts.lean_score, ts.observation_count, ts.last_observed "
        "FROM tensions t "
        "JOIN tension_states ts ON ts.tension_id = t.id "
        "ORDER BY t.name"
    )
    if not rows:
        return []

    result = []
    for row in rows.split("\n"):
        row = row.strip()
        if not row:
            continue
        parts = row.split("|")
        if len(parts) < 7:
            continue
        lean = float(parts[4])
        result.append({
            "name": parts[0],
            "pole_a": parts[1],
            "pole_b": parts[2],
            "description": parts[3],
            "lean_score": lean,
            "observation_count": int(parts[5]),
            "last_observed": parts[6] or None,
            "interpretation": _interpret_lean(parts[1], parts[2], lean),
        })
    return result


def _interpret_lean(pole_a: str, pole_b: str, lean: float) -> str:
    """Generate a human-readable interpretation of lean_score."""
    if lean < 0.2:
        return f"Strongly leans toward {pole_a}"
    elif lean < 0.35:
        return f"Leans toward {pole_a}"
    elif lean < 0.45:
        return f"Slightly leans toward {pole_a}"
    elif lean <= 0.55:
        return f"Balanced between {pole_a} and {pole_b}"
    elif lean <= 0.65:
        return f"Slightly leans toward {pole_b}"
    elif lean <= 0.8:
        return f"Leans toward {pole_b}"
    else:
        return f"Strongly leans toward {pole_b}"


def get_tension_context() -> str:
    """Format tension states for prompt injection."""
    tensions = get_all_tensions()
    if not tensions:
        return ""

    lines = ["## Developmental Tensions"]
    for t in tensions:
        obs = t["observation_count"]
        if obs == 0:
            lines.append(f"- **{t['pole_a']} vs {t['pole_b']}**: No observations yet")
        else:
            score_bar = _score_bar(t["lean_score"])
            lines.append(
                f"- **{t['pole_a']} vs {t['pole_b']}**: {score_bar} "
                f"({t['interpretation']}, {obs} observations)"
            )
    return "\n".join(lines)


def _score_bar(score: float) -> str:
    """Visual bar: [A ====|==== B] where | shows the lean position."""
    total = 20
    pos = int(score * total)
    pos = max(0, min(total, pos))
    return "[" + "=" * pos + "|" + "=" * (total - pos) + "]"


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  tension_tracker.py seed")
        print("  tension_tracker.py observe <tension_name> <pole_a|pole_b|balanced> [context] [confidence]")
        print("  tension_tracker.py status [tension_name]")
        print("  tension_tracker.py context")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "seed":
        seed_tensions()

    elif cmd == "observe":
        if len(sys.argv) < 4:
            print("Usage: tension_tracker.py observe <tension_name> <resolution> [context] [confidence]")
            sys.exit(1)
        name = sys.argv[2]
        resolution = sys.argv[3]
        context = sys.argv[4] if len(sys.argv) > 4 else ""
        confidence = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7
        observe_tension(name, resolution, context, confidence)

    elif cmd == "status":
        if len(sys.argv) > 2:
            state = get_tension_state(sys.argv[2])
            if state:
                print(f"Tension: {state['pole_a']} vs {state['pole_b']}")
                print(f"  Lean score: {state['lean_score']:.3f}")
                print(f"  {state['interpretation']}")
                print(f"  Observations: {state['observation_count']}")
                print(f"  Last observed: {state['last_observed'] or 'never'}")
            else:
                print(f"Tension '{sys.argv[2]}' not found")
        else:
            tensions = get_all_tensions()
            if not tensions:
                print("No tensions found. Run 'seed' first.")
            for t in tensions:
                bar = _score_bar(t["lean_score"])
                print(f"{t['name']}: {bar} {t['interpretation']} ({t['observation_count']} obs)")

    elif cmd == "context":
        print(get_tension_context())

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
