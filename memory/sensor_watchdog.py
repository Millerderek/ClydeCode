#!/usr/bin/env python3
"""
sensor_watchdog.py — Home Assistant Sensor Health Monitor

Phase 4+ of OpenClaw Cognitive Architecture.
Detects stale sensors, counter anomalies, and phantom template outputs
from Home Assistant, storing findings as awareness signals.

Cron: */30 * * * *

Usage:
    python3 sensor_watchdog.py              # Full scan
    python3 sensor_watchdog.py --dry-run    # Print findings without storing
"""

import json
import os
import sys
import argparse
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# HA credentials
_ha_env = Path("/root/APIKeys/ha_token.env")
HA_URL = ""
HA_TOKEN = ""

if _ha_env.exists():
    for line in _ha_env.read_text().splitlines():
        line = line.strip()
        if line.startswith("HA_URL="):
            HA_URL = line.split("=", 1)[1]
        elif line.startswith("HA_TOKEN="):
            HA_TOKEN = line.split("=", 1)[1]

# Sensors to monitor for staleness (entity_id → max age in minutes)
STALE_THRESHOLDS = {
    # Shelly Pro 3EM — updates every ~30s
    "sensor.shelly_grid_energy_import_raw": 10,
    "sensor.shelly_grid_energy_export_raw": 10,
    "sensor.shelly_grid_l1_power": 10,
    "sensor.shelly_grid_l2_power": 10,
    "sensor.shelly_grid_total_power": 10,
    # Solar inverter telemetry — ANJ-12KP via ESP32 bridge
    "sensor.anj_12kp_inverter_bridge_pv_input_power": 15,
    "sensor.anj_12kp_inverter_bridge_inverter_mode": 15,
    # SolarEdge production
    "sensor.solaredge_i1_ac_power": 30,
    "sensor.solaredge_i1_ac_energy": 30,
    # Template sensors — should update when source updates
    "sensor.shelly_grid_total_energy_import": 15,
    "sensor.shelly_grid_total_energy_export": 15,
    "sensor.lightning_charger_power": 30,
}

# Counter sensors to track for flip-flops and anomalies
COUNTER_SENSORS = [
    "sensor.shelly_grid_energy_import_raw",
    "sensor.shelly_grid_energy_export_raw",
]

# Template sensors with known phantom-output risk
PHANTOM_SENSORS = {
    "sensor.lightning_charger_power": {
        "source_entity": "switch.emporia_vue_charger_emporia_vue_574311",
        "zero_when_off": True,
    },
}

# Redis key prefix for counter state tracking
REDIS_PREFIX = "sensor_watchdog:"
REDIS_TTL = 7200  # 2 hours

# Telegram notification
TELEGRAM_SCRIPT = "/root/.openclaw/memory/telegram_notify.py"

# ═══════════════════════════════════════════════════════════════════════════════
# HA API Client
# ═══════════════════════════════════════════════════════════════════════════════

def _ha_get(path: str) -> dict | list | None:
    """GET request to HA REST API."""
    import urllib.request
    import urllib.error

    url = f"{HA_URL}/api/{path}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        print(f"  [warn] HA API error ({path}): {e}", file=sys.stderr)
        return None


def _get_sensor_state(entity_id: str) -> dict | None:
    """Fetch a single entity state from HA."""
    return _ha_get(f"states/{entity_id}")


def _get_all_sensor_states(entity_ids: list[str]) -> dict[str, dict]:
    """Fetch states for multiple entities. Returns {entity_id: state_dict}."""
    all_states = _ha_get("states")
    if not all_states:
        return {}
    id_set = set(entity_ids)
    return {s["entity_id"]: s for s in all_states if s["entity_id"] in id_set}


# ═══════════════════════════════════════════════════════════════════════════════
# Redis State Tracking
# ═══════════════════════════════════════════════════════════════════════════════

def _get_redis():
    """Get Redis connection via db.py."""
    try:
        from db import get_redis
        return get_redis()
    except Exception:
        return None


def _redis_get_counter(entity_id: str) -> dict | None:
    """Get last known counter state from Redis."""
    r = _get_redis()
    if not r:
        return None
    key = f"{REDIS_PREFIX}{entity_id}"
    raw = r.get(key)
    if raw:
        return json.loads(raw)
    return None


def _redis_set_counter(entity_id: str, value: float, timestamp: str):
    """Store current counter state in Redis."""
    r = _get_redis()
    if not r:
        return
    key = f"{REDIS_PREFIX}{entity_id}"
    r.set(key, json.dumps({
        "value": value,
        "timestamp": timestamp,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }), ex=REDIS_TTL)


# ═══════════════════════════════════════════════════════════════════════════════
# Signal Storage (reuses peripheral_awareness.store_signal)
# ═══════════════════════════════════════════════════════════════════════════════

def _store_signal(signal_type: str, title: str, description: str,
                  confidence: float = 0.7, priority: str = "low",
                  related_entities: list[str] | None = None,
                  expires_hours: int = 48) -> int | None:
    """Store an awareness signal via peripheral_awareness."""
    try:
        from peripheral_awareness import store_signal
        return store_signal(
            signal_type=signal_type,
            title=title,
            description=description,
            confidence=confidence,
            priority=priority,
            related_entities=related_entities,
            source_query="sensor_watchdog",
            expires_hours=expires_hours,
        )
    except Exception as e:
        print(f"  [warn] Failed to store signal: {e}", file=sys.stderr)
        return None


def _notify(title: str, body: str):
    """Send Telegram notification."""
    try:
        subprocess.run(
            [sys.executable, TELEGRAM_SCRIPT, f"🔍 Sensor Watchdog: {title}\n\n{body}"],
            capture_output=True, timeout=15,
        )
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Detector 1: Stale Sensors
# ═══════════════════════════════════════════════════════════════════════════════

def detect_stale_sensors(states: dict[str, dict], dry_run: bool = False) -> list[dict]:
    """Detect sensors that haven't updated within their expected window."""
    findings = []
    now = datetime.now(timezone.utc)

    for entity_id, max_age_min in STALE_THRESHOLDS.items():
        state = states.get(entity_id)
        if not state:
            findings.append({
                "type": "missing_sensor",
                "entity_id": entity_id,
                "message": f"Sensor not found in HA",
                "priority": "medium",
            })
            continue

        if state.get("state") in ("unavailable", "unknown"):
            findings.append({
                "type": "stale_sensor",
                "entity_id": entity_id,
                "message": f"Sensor state is '{state['state']}'",
                "priority": "medium",
            })
            continue

        last_changed = state.get("last_changed") or state.get("last_updated")
        if not last_changed:
            continue

        try:
            # HA returns ISO 8601 timestamps
            ts = datetime.fromisoformat(last_changed.replace("Z", "+00:00"))
            age_min = (now - ts).total_seconds() / 60

            if age_min > max_age_min:
                findings.append({
                    "type": "stale_sensor",
                    "entity_id": entity_id,
                    "message": f"Last updated {age_min:.0f}m ago (threshold: {max_age_min}m)",
                    "priority": "low" if age_min < max_age_min * 3 else "medium",
                    "age_minutes": round(age_min),
                })
        except (ValueError, TypeError):
            continue

    if not dry_run:
        for f in findings:
            _store_signal(
                signal_type="anomaly",
                title=f"Stale: {f['entity_id'].split('.')[-1]}",
                description=f["message"],
                confidence=0.8,
                priority=f["priority"],
                related_entities=[f["entity_id"], "Home Assistant"],
                expires_hours=2,  # Short TTL — next scan replaces
            )

    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# Detector 2: Counter Anomalies
# ═══════════════════════════════════════════════════════════════════════════════

def detect_counter_anomalies(states: dict[str, dict], dry_run: bool = False) -> list[dict]:
    """Detect counter flip-flops and unexpected resets."""
    findings = []

    for entity_id in COUNTER_SENSORS:
        state = states.get(entity_id)
        if not state or state.get("state") in ("unavailable", "unknown"):
            continue

        try:
            current_val = float(state["state"])
        except (ValueError, TypeError):
            continue

        timestamp = state.get("last_changed", "")
        prev = _redis_get_counter(entity_id)

        if prev:
            prev_val = prev["value"]
            delta = current_val - prev_val

            # Flip-flop: value dropped significantly then came back
            # This was the root cause of the inflated TOU meters
            if delta < -1.0:
                # Counter went backwards — possible flip-flop
                pct_drop = abs(delta) / max(prev_val, 0.01) * 100
                findings.append({
                    "type": "counter_drop",
                    "entity_id": entity_id,
                    "message": (
                        f"Counter dropped from {prev_val:.3f} to {current_val:.3f} "
                        f"(Δ={delta:.3f} kWh, {pct_drop:.0f}% drop)"
                    ),
                    "priority": "high" if pct_drop > 20 else "medium",
                    "delta": round(delta, 3),
                })
            elif delta > 50:
                # Unreasonably large jump (50 kWh in 30 minutes)
                findings.append({
                    "type": "counter_spike",
                    "entity_id": entity_id,
                    "message": (
                        f"Counter jumped from {prev_val:.3f} to {current_val:.3f} "
                        f"(Δ={delta:.3f} kWh in ~30 min)"
                    ),
                    "priority": "high",
                    "delta": round(delta, 3),
                })

        # Always update Redis with current value
        _redis_set_counter(entity_id, current_val, timestamp)

    if not dry_run:
        for f in findings:
            sig_id = _store_signal(
                signal_type="anomaly",
                title=f"Counter: {f['entity_id'].split('.')[-1]}",
                description=f["message"],
                confidence=0.9,
                priority=f["priority"],
                related_entities=[f["entity_id"], "Shelly Pro 3EM", "Home Assistant"],
                expires_hours=6,
            )
            if f["priority"] == "high":
                _notify(
                    f"Counter Anomaly — {f['entity_id'].split('.')[-1]}",
                    f["message"],
                )

    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# Detector 3: Phantom Template Outputs
# ═══════════════════════════════════════════════════════════════════════════════

def detect_phantom_outputs(states: dict[str, dict], dry_run: bool = False) -> list[dict]:
    """Detect template sensors reporting non-zero when source is inactive."""
    findings = []

    for entity_id, config in PHANTOM_SENSORS.items():
        state = states.get(entity_id)
        source_state = states.get(config["source_entity"])

        if not state or not source_state:
            continue

        try:
            sensor_val = float(state["state"])
        except (ValueError, TypeError):
            continue

        source_is_off = source_state.get("state") in ("off", "unavailable", "unknown")

        # Check source attributes for deeper status
        source_attrs = source_state.get("attributes", {})
        source_status = source_attrs.get("status", "")
        source_inactive = source_status in ("Standby", "Off", "Disabled", "")

        if config.get("zero_when_off"):
            # Sensor should be 0 when source is off or inactive
            if (source_is_off or source_inactive) and sensor_val > 0:
                findings.append({
                    "type": "phantom_output",
                    "entity_id": entity_id,
                    "message": (
                        f"Reporting {sensor_val}W but source "
                        f"'{config['source_entity'].split('.')[-1]}' "
                        f"is {'off' if source_is_off else f'status={source_status}'}"
                    ),
                    "priority": "medium",
                    "source_entity": config["source_entity"],
                })

    if not dry_run:
        for f in findings:
            _store_signal(
                signal_type="anomaly",
                title=f"Phantom: {f['entity_id'].split('.')[-1]}",
                description=f["message"],
                confidence=0.85,
                priority=f["priority"],
                related_entities=[
                    f["entity_id"],
                    f.get("source_entity", ""),
                    "Home Assistant",
                ],
                expires_hours=4,
            )

    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# Detector 4: Session Size Auto-Rotate
# ═══════════════════════════════════════════════════════════════════════════════

SESSIONS_JSON = Path("/root/.openclaw/agents/main/sessions/sessions.json")
SESSION_SIZE_LIMIT = 1.5 * 1024 * 1024  # 1.5 MB
SESSION_COOLDOWN_KEY = f"{REDIS_PREFIX}session_rotate_cooldown"
SESSION_COOLDOWN_TTL = 600  # 10 min — prevent rapid restart loops


def _get_active_session_file() -> Path | None:
    """Find the active Telegram session JSONL for Derek."""
    if not SESSIONS_JSON.exists():
        return None
    try:
        data = json.loads(SESSIONS_JSON.read_text())
        tg = data.get("agent:main:telegram:direct:8260442678", {})
        sf = tg.get("sessionFile")
        if sf:
            p = Path(sf)
            if p.exists():
                return p
    except Exception:
        pass
    return None


def _run_session_ingest() -> bool:
    """Trigger session ingest to capture recent facts before rotation."""
    try:
        result = subprocess.run(
            [sys.executable, "/root/ClydeMemory/session_ingest.py"],
            capture_output=True, text=True, timeout=120,
        )
        print(f"  Session ingest: rc={result.returncode}")
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines()[-5:]:
                print(f"    {line}")
        return result.returncode == 0
    except Exception as e:
        print(f"  [warn] Session ingest failed: {e}", file=sys.stderr)
        return False


def _restart_bot() -> bool:
    """Gracefully restart the Telegram bot (SIGTERM → screen/Docker restarts it)."""
    import signal
    try:
        result = subprocess.run(
            ["pgrep", "-f", "python3 clydecodebot.py"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [int(p.strip()) for p in result.stdout.strip().splitlines() if p.strip()]
        if not pids:
            print("  [warn] Bot process not found — skipping restart")
            return False

        bot_pid = pids[0]
        print(f"  Sending SIGTERM to bot PID {bot_pid}...")
        os.kill(bot_pid, signal.SIGTERM)
        return True
    except Exception as e:
        print(f"  [warn] Bot restart failed: {e}", file=sys.stderr)
        return False


def detect_session_bloat(dry_run: bool = False) -> list[dict]:
    """Detect oversized session JSONL and auto-rotate if needed."""
    findings = []

    session_file = _get_active_session_file()
    if not session_file:
        print("  No active Telegram session found.")
        return findings

    size = session_file.stat().st_size
    size_mb = size / (1024 * 1024)
    line_count = sum(1 for _ in open(session_file, "rb"))

    print(f"  Active session: {session_file.name}")
    print(f"  Size: {size_mb:.1f} MB ({line_count} lines)")

    if size < SESSION_SIZE_LIMIT:
        print(f"  ✓ Under limit ({SESSION_SIZE_LIMIT / 1024 / 1024:.1f} MB)")
        return findings

    findings.append({
        "type": "session_bloat",
        "entity_id": f"session:{session_file.stem}",
        "message": f"Session {size_mb:.1f} MB / {line_count} lines (limit: {SESSION_SIZE_LIMIT / 1024 / 1024:.1f} MB)",
        "priority": "medium",
        "size_mb": round(size_mb, 1),
        "line_count": line_count,
    })

    if dry_run:
        print(f"  ⚠ Would rotate session ({size_mb:.1f} MB)")
        return findings

    # Check cooldown — prevent restart loops
    r = _get_redis()
    if r and r.get(SESSION_COOLDOWN_KEY):
        print(f"  ⏳ Cooldown active — skipping rotation")
        return findings

    # Set cooldown FIRST to prevent loops even if restart fails
    if r:
        r.set(SESSION_COOLDOWN_KEY, "1", ex=SESSION_COOLDOWN_TTL)

    print(f"\n  🔄 Session auto-rotate triggered ({size_mb:.1f} MB)")

    # Step 1: Run session ingest to capture recent facts
    print("  Step 1: Running session ingest...")
    _run_session_ingest()

    # Step 2: Store awareness signal
    _store_signal(
        signal_type="maintenance",
        title=f"Session rotated ({size_mb:.1f} MB)",
        description=f"Auto-rotated Telegram session at {size_mb:.1f} MB / {line_count} lines. "
                     f"Session ingest ran before restart.",
        confidence=1.0,
        priority="low",
        related_entities=["OpenClaw", "Telegram Bot", "Clyde"],
        expires_hours=24,
    )

    # Step 3: Notify Derek
    _notify(
        "Session Auto-Rotate",
        f"Session hit {size_mb:.1f} MB ({line_count} lines).\n"
        f"Ingested recent facts and restarting bot — back in ~30s.",
    )

    # Step 4: Restart bot
    print("  Step 2: Restarting bot...")
    _restart_bot()

    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# Main Scan
# ═══════════════════════════════════════════════════════════════════════════════

def run_scan(dry_run: bool = False) -> dict:
    """Run all detectors and return summary."""
    print("=" * 60)
    print(f"Sensor Watchdog — Scan")
    print(f"  Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"  HA:   {HA_URL}")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print("=" * 60)

    if not HA_URL or not HA_TOKEN:
        print("[error] HA credentials not found at /root/APIKeys/ha_token.env")
        return {"error": "no_credentials"}

    # Gather all entity IDs we need
    all_ids = set(STALE_THRESHOLDS.keys())
    all_ids.update(COUNTER_SENSORS)
    all_ids.update(PHANTOM_SENSORS.keys())
    for cfg in PHANTOM_SENSORS.values():
        all_ids.add(cfg["source_entity"])

    # Single bulk fetch
    print(f"\nFetching {len(all_ids)} entity states from HA...")
    states = _get_all_sensor_states(list(all_ids))
    print(f"  Got {len(states)} states back.")

    if not states:
        print("[error] No states returned from HA API.")
        return {"error": "no_states"}

    # Run detectors
    print("\n--- Stale Sensors ---")
    stale = detect_stale_sensors(states, dry_run=dry_run)
    for f in stale:
        print(f"  ⚠ {f['entity_id']}: {f['message']}")
    if not stale:
        print("  ✓ All sensors fresh.")

    print("\n--- Counter Anomalies ---")
    counters = detect_counter_anomalies(states, dry_run=dry_run)
    for f in counters:
        marker = "🔴" if f["priority"] == "high" else "⚠"
        print(f"  {marker} {f['entity_id']}: {f['message']}")
    if not counters:
        print("  ✓ All counters healthy.")

    print("\n--- Phantom Outputs ---")
    phantoms = detect_phantom_outputs(states, dry_run=dry_run)
    for f in phantoms:
        print(f"  ⚠ {f['entity_id']}: {f['message']}")
    if not phantoms:
        print("  ✓ No phantom outputs.")

    print("\n--- Session Size ---")
    session = detect_session_bloat(dry_run=dry_run)
    if not session:
        pass  # Status already printed inside detector

    all_findings = stale + counters + phantoms + session
    total_findings = len(all_findings)
    high_prio = sum(1 for f in all_findings if f.get("priority") == "high")

    print(f"\n{'=' * 60}")
    print(f"Summary: {total_findings} findings ({high_prio} high priority)")
    print(f"{'=' * 60}")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stale": len(stale),
        "counter_anomalies": len(counters),
        "phantom_outputs": len(phantoms),
        "session_bloat": len(session),
        "total": total_findings,
        "high_priority": high_prio,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensor Watchdog — HA Sensor Health Monitor")
    parser.add_argument("--dry-run", action="store_true", help="Print findings without storing")
    args = parser.parse_args()

    result = run_scan(dry_run=args.dry_run)
    if result.get("error"):
        sys.exit(1)
