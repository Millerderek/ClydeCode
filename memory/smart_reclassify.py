#!/usr/bin/env python3
"""
smart_reclassify.py — Name-pattern + relationship reclassification for general entities.

Combines keyword matching on entity names with relationship-based signals.
Filters out ambiguous/generic entities that shouldn't be reclassified.
"""

import re
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

# ── Name-based classification patterns ──────────────────────────────────
# Order matters: first match wins

NAME_PATTERNS = [
    # Credentials
    (r'(?i)\b(api.?key|token|credentials?|password|secret|auth.?token)\b', 'credential'),

    # Registers (Modbus, hardware)
    (r'(?i)\b(register\s+0x|reg\s+0x|0x[0-9a-fA-F]{3,})\b', 'register'),
    (r'(?i)^Register\s+', 'register'),

    # Infrastructure (IPs, CIDRs, network)
    (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(/\d{1,2})?\b', 'infrastructure'),
    (r'(?i)\b(subnet|CIDR|VLAN|gateway|firewall|DNS|DHCP|NAT|DNAT|VPN|WireGuard)\b', 'infrastructure'),
    (r'(?i)\b(iptables|ip rule|network)\b', 'infrastructure'),
    (r'(?i)\b(routing)\b(?!.*\b(model|config|task|template))', 'infrastructure'),

    # Containers
    (r'(?i)\b(container|docker)\b', 'container'),

    # Sensors
    (r'(?i)\b(sensor|temperature|humidity|power meter|energy meter|Shelly)\b', 'sensor'),

    # Protocols
    (r'(?i)^(MQTT|Modbus|SIP|HTTP|REST|gRPC|RS485|RS232|TCP|UDP|WebSocket)$', 'protocol'),

    # Automation
    (r'(?i)\b(automation|schedule|cron|timer|trigger)\b', 'automation'),

    # Scripts / config files
    (r'(?i)\.(py|sh|yaml|yml|json|conf|toml|ini)$', 'script'),
    (r'(?i)^(script|config|Dockerfile)', 'script'),

    # People (emails)
    (r'@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 'person'),

    # Agents / bots (exclude "config" / "credential" combos already caught above)
    (r'(?i)\b(bot|agent|daemon|worker)\b', 'agent'),
    (r'(?i)\bbridge\b(?!.*\bconfig)', 'agent'),

    # Devices
    (r'(?i)\b(ESP32|inverter|dongle|relay|board|hardware|firmware)\b', 'device'),

    # Services
    (r'(?i)\b(API|endpoint|server|service|gateway|webhook|dashboard)\b', 'service'),

    # Systems / platforms
    (r'(?i)^(Home Assistant|HAOS|Node-RED|Mosquitto|Grafana|PostgreSQL|Redis|Qdrant|Mem0)$', 'system'),
]

# ── Entities to skip (too generic, ambiguous, or just values) ───────────
SKIP_PATTERNS = [
    r'^.{0,2}$',                     # Too short
    r'^\d+$',                        # Pure numbers
    r'^[A-Z]{1,3}$',                 # Abbreviations too short
    r'(?i)^(true|false|none|null)$', # Boolean-like
    r'(?i)^(yes|no|on|off)$',
    r'(?i)^(Project|project)$',      # Too generic
    r'(?i)^(traffic|core tools|data|info|status|error|output|input|result)$',
    r'(?i)^(Phase \d|Step \d)',      # Process steps, not entities
]


def classify_by_name(name: str) -> str | None:
    """Try to classify an entity by its name using patterns. Returns type or None."""
    # Check skip patterns first
    for pat in SKIP_PATTERNS:
        if re.search(pat, name):
            return None  # Skip this entity

    # Try classification patterns
    for pat, entity_type in NAME_PATTERNS:
        if re.search(pat, name):
            return entity_type

    return None


def get_relationship_signal(entity_id: int) -> str | None:
    """Get classification signal from relationships (refined heuristic)."""
    result = db.pg_query(
        "SELECT r.predicate, "
        "CASE WHEN r.source_id = %s THEN 'outbound' ELSE 'inbound' END as direction, "
        "CASE WHEN r.source_id = %s THEN e2.type ELSE e2.type END as connected_type "
        "FROM relationships r "
        "JOIN entities e2 ON (e2.id = CASE WHEN r.source_id = %s THEN r.target_id ELSE r.source_id END) "
        "WHERE r.source_id = %s OR r.target_id = %s "
        "LIMIT 20;",
        (entity_id, entity_id, entity_id, entity_id, entity_id)
    )
    if not result:
        return None

    predicates = []
    connected_types = []
    directions = []
    for line in result.split("\n"):
        if "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) >= 3:
            predicates.append(parts[0].strip())
            directions.append(parts[1].strip())
            connected_types.append(parts[2].strip())

    # Strong signals only
    pred_set = set(predicates)

    if "READS_FROM" in pred_set or "WRITES_TO" in pred_set:
        if "register" in connected_types:
            return "device"  # reads/writes registers = device
        return "service"

    if "PUBLISHES_TO" in pred_set or "SUBSCRIBES_TO" in pred_set:
        return "service"

    if "MONITORS" in pred_set and any(
        d == "outbound" for p, d in zip(predicates, directions) if p == "MONITORS"
    ):
        return "agent"

    if "RUNS_ON" in pred_set:
        # Check direction: if it runs on infra, it's a service/script
        for p, d, ct in zip(predicates, directions, connected_types):
            if p == "RUNS_ON" and d == "outbound" and ct in ("infrastructure", "device"):
                return "service"

    return None


def smart_reclassify(dry_run: bool = True, verbose: bool = False) -> tuple[list, int]:
    """
    Reclassify general entities using name patterns + relationship signals.

    Returns list of (entity_id, name, new_type, reason) tuples.
    """
    # Get all general entities
    result = db.pg_query(
        "SELECT id, name FROM entities WHERE type = 'general' ORDER BY id;"
    )
    if not result:
        return []

    suggestions = []
    skipped = 0

    for line in result.split("\n"):
        if "|" not in line:
            continue
        parts = line.split("|", 1)
        if len(parts) < 2:
            continue
        try:
            eid = int(parts[0])
        except ValueError:
            continue
        name = parts[1].strip()

        # Try name-based classification first (higher confidence)
        name_type = classify_by_name(name)
        if name_type is None:
            # Check if it was skipped by skip patterns
            should_skip = any(re.search(pat, name) for pat in SKIP_PATTERNS)
            if should_skip:
                skipped += 1
                continue

        # Try relationship-based signal
        rel_type = get_relationship_signal(eid)

        # Decide final type
        final_type = None
        reason = ""

        if name_type and rel_type and name_type == rel_type:
            # Both agree — high confidence
            final_type = name_type
            reason = f"name pattern + relationship agree → {name_type}"
        elif name_type:
            # Name pattern only — medium confidence
            final_type = name_type
            reason = f"name pattern match"
        elif rel_type:
            # Relationship only — lower confidence, only apply for strong signals
            if verbose:
                print(f"  [rel-only] [{eid}] {name} → {rel_type} (skipping, name unclear)")
            continue  # Skip relationship-only — too many false positives

        if final_type:
            suggestions.append((eid, name, final_type, reason))
            if not dry_run:
                db.pg_execute(
                    "UPDATE entities SET type = %s, updated_at = NOW() WHERE id = %s;",
                    (final_type, eid)
                )

    return suggestions, skipped


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    CYAN = "\033[96m"
    GREEN = "\033[92m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    suggestions, skipped = smart_reclassify(dry_run=not args.apply, verbose=args.verbose)

    if not suggestions:
        print("  No reclassification suggestions")
    else:
        action = "Applied" if args.apply else "Suggested"
        print(f"\n  {action} {len(suggestions)} reclassifications ({skipped} skipped as too generic):\n")

        # Group by target type
        by_type = defaultdict(list)
        for eid, name, new_type, reason in suggestions:
            by_type[new_type].append((eid, name, reason))

        for typ in sorted(by_type.keys()):
            items = by_type[typ]
            print(f"  {GREEN}→ {typ}{RESET} ({len(items)} entities)")
            for eid, name, reason in items[:8]:
                print(f"    [{eid}] {name} {DIM}({reason}){RESET}")
            if len(items) > 8:
                print(f"    ... and {len(items) - 8} more")
            print()
