#!/usr/bin/env python3
"""
soul_growth.py — Identity Development Over Time for OpenClaw.

Implements the preference maturation ladder, time-layered identity management,
and reflection automation from soul.md's Growth Through Time framework.

Phase 6 of OpenClaw Cognitive Architecture.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

MATURITY_LABELS = {
    0: "observation",
    1: "leaning",
    2: "soft_preference",
    3: "stable_preference",
    4: "trait_candidate",
}

# Promotion thresholds: (min_supports, max_challenge_rate, min_days_span)
PROMOTION_THRESHOLDS = {
    0: (3, 0.30, 0),       # 0→1: 3+ supporting, <30% challenge rate
    1: (5, 0.25, 0),       # 1→2: 5+ supporting, <25% challenge rate
    2: (8, 0.20, 7),       # 2→3: 8+ supporting over 7+ days, <20% challenge rate
    3: (12, 0.15, 30),     # 3→4: 12+ supporting over 30+ days, <15% challenge rate
}

# Demotion: if challenge rate exceeds this, demote one level
DEMOTION_THRESHOLD = 0.40

# OpenRouter config
_key_file = os.path.expanduser("~/APIKeys/openrouter.env")
OPENROUTER_API_KEY = ""
if os.path.exists(_key_file):
    with open(_key_file) as f:
        for line in f:
            if line.startswith("OPENROUTER_API_KEY="):
                OPENROUTER_API_KEY = line.strip().split("=", 1)[1]
OPENROUTER_MODEL = "anthropic/claude-haiku-4.5"

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _log(msg: str):
    print(f"[soul_growth] {msg}", flush=True)


def _now_utc():
    return datetime.now(timezone.utc)


# ═══════════════════════════════════════════════════════════════════════════════
# Schema Creation
# ═══════════════════════════════════════════════════════════════════════════════

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS preferences (
    id SERIAL PRIMARY KEY,
    domain TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    maturity INTEGER NOT NULL DEFAULT 0,
    evidence_count INTEGER DEFAULT 1,
    first_observed TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_confirmed TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_challenged TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(domain, title)
);

CREATE TABLE IF NOT EXISTS preference_evidence (
    id SERIAL PRIMARY KEY,
    preference_id INTEGER NOT NULL REFERENCES preferences(id) ON DELETE CASCADE,
    evidence_type TEXT NOT NULL DEFAULT 'supports',
    description TEXT NOT NULL,
    source TEXT DEFAULT 'auto',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS identity_layers (
    id SERIAL PRIMARY KEY,
    layer TEXT NOT NULL,
    attribute TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    UNIQUE(layer, attribute)
);

CREATE TABLE IF NOT EXISTS reflection_log (
    id SERIAL PRIMARY KEY,
    reflection_type TEXT NOT NULL,
    content TEXT NOT NULL,
    insights TEXT[],
    preference_updates INTEGER DEFAULT 0,
    tension_observations INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_preferences_domain ON preferences(domain);
CREATE INDEX IF NOT EXISTS idx_preferences_maturity ON preferences(maturity);
CREATE INDEX IF NOT EXISTS idx_preferences_active ON preferences(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_identity_layer ON identity_layers(layer);
CREATE INDEX IF NOT EXISTS idx_reflection_type ON reflection_log(reflection_type);
"""


def create_schema():
    """Create all Phase 6 tables and indexes."""
    _log("Creating schema...")
    db.pg_execute_many(SCHEMA_SQL)
    _log("Schema created.")


# ═══════════════════════════════════════════════════════════════════════════════
# Preference Management
# ═══════════════════════════════════════════════════════════════════════════════

def add_preference(domain: str, title: str, description: str, initial_maturity: int = 0) -> int:
    """Add a new preference. Returns its id."""
    result = db.pg_query(
        "INSERT INTO preferences (domain, title, description, maturity) "
        "VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (domain, title) DO UPDATE SET "
        "  description = EXCLUDED.description, "
        "  maturity = GREATEST(preferences.maturity, EXCLUDED.maturity) "
        "RETURNING id",
        (domain, title, description, initial_maturity)
    )
    if result:
        return int(result.strip())
    return -1


def _get_evidence_stats(preference_id: int) -> dict:
    """Get supporting/challenging counts and date range for a preference."""
    result = db.pg_query(
        "SELECT "
        "  COUNT(*) FILTER (WHERE evidence_type = 'supports') as supports, "
        "  COUNT(*) FILTER (WHERE evidence_type = 'challenges') as challenges, "
        "  MIN(created_at) as first_evidence, "
        "  MAX(created_at) as last_evidence "
        "FROM preference_evidence "
        "WHERE preference_id = %s",
        (preference_id,)
    )
    if not result:
        return {"supports": 0, "challenges": 0, "days_span": 0, "challenge_rate": 0.0}

    parts = result.split("|")
    supports = int(parts[0]) if parts[0] else 0
    challenges = int(parts[1]) if parts[1] else 0
    total = supports + challenges
    challenge_rate = challenges / total if total > 0 else 0.0

    # Calculate days span
    days_span = 0
    if parts[2] and parts[3]:
        try:
            first = datetime.fromisoformat(parts[2].replace("+00", "+00:00"))
            last = datetime.fromisoformat(parts[3].replace("+00", "+00:00"))
            days_span = (last - first).days
        except Exception:
            pass

    return {
        "supports": supports,
        "challenges": challenges,
        "challenge_rate": challenge_rate,
        "days_span": days_span,
    }


def confirm_preference(preference_id: int, evidence_description: str, source: str = "auto"):
    """Add supporting evidence. Increment count and check for promotion."""
    db.pg_execute(
        "INSERT INTO preference_evidence (preference_id, evidence_type, description, source) "
        "VALUES (%s, 'supports', %s, %s)",
        (preference_id, evidence_description, source)
    )
    db.pg_execute(
        "UPDATE preferences "
        "SET evidence_count = evidence_count + 1, last_confirmed = NOW() "
        "WHERE id = %s",
        (preference_id,)
    )
    # Check if promotion is warranted
    _try_promote(preference_id)


def challenge_preference(preference_id: int, evidence_description: str, source: str = "auto"):
    """Add challenging evidence. Check for demotion."""
    db.pg_execute(
        "INSERT INTO preference_evidence (preference_id, evidence_type, description, source) "
        "VALUES (%s, 'challenges', %s, %s)",
        (preference_id, evidence_description, source)
    )
    db.pg_execute(
        "UPDATE preferences SET last_challenged = NOW() WHERE id = %s",
        (preference_id,)
    )
    # Check if demotion is warranted
    _try_demote(preference_id)


def _try_promote(preference_id: int):
    """Attempt to promote a preference up the maturity ladder."""
    result = db.pg_query("SELECT maturity FROM preferences WHERE id = %s", (preference_id,))
    if not result:
        return
    current = int(result.strip())
    if current >= 4:
        return  # Already at max

    threshold = PROMOTION_THRESHOLDS.get(current)
    if not threshold:
        return

    min_supports, max_challenge_rate, min_days = threshold
    stats = _get_evidence_stats(preference_id)

    if (stats["supports"] >= min_supports and
            stats["challenge_rate"] < max_challenge_rate and
            stats["days_span"] >= min_days):
        promote_preference(preference_id)


def _try_demote(preference_id: int):
    """Attempt to demote a preference if challenge rate is too high."""
    result = db.pg_query("SELECT maturity FROM preferences WHERE id = %s", (preference_id,))
    if not result:
        return
    current = int(result.strip())
    if current <= 0:
        return

    stats = _get_evidence_stats(preference_id)
    if stats["challenge_rate"] > DEMOTION_THRESHOLD:
        demote_preference(preference_id)


def promote_preference(preference_id: int):
    """Move a preference up one maturity level."""
    result = db.pg_query("SELECT maturity, domain, title FROM preferences WHERE id = %s", (preference_id,))
    if not result:
        return
    parts = result.split("|")
    current = int(parts[0])
    if current >= 4:
        return
    new_level = current + 1
    db.pg_execute("UPDATE preferences SET maturity = %s WHERE id = %s", (new_level, preference_id))
    _log(f"Promoted {parts[1]}/{parts[2]}: {MATURITY_LABELS[current]} -> {MATURITY_LABELS[new_level]}")


def demote_preference(preference_id: int):
    """Move a preference down one maturity level."""
    result = db.pg_query("SELECT maturity, domain, title FROM preferences WHERE id = %s", (preference_id,))
    if not result:
        return
    parts = result.split("|")
    current = int(parts[0])
    if current <= 0:
        return
    new_level = current - 1
    db.pg_execute("UPDATE preferences SET maturity = %s WHERE id = %s", (new_level, preference_id))
    _log(f"Demoted {parts[1]}/{parts[2]}: {MATURITY_LABELS[current]} -> {MATURITY_LABELS[new_level]}")


def get_preferences(domain: str = None, min_maturity: int = 0) -> list:
    """Get active preferences, optionally filtered."""
    clauses = ["is_active = TRUE"]
    params = []
    if domain:
        clauses.append("domain = %s")
        params.append(domain)
    if min_maturity > 0:
        clauses.append("maturity >= %s")
        params.append(min_maturity)
    where = " AND ".join(clauses)

    result = db.pg_query(
        "SELECT id, domain, title, description, maturity, evidence_count, "
        "       first_observed, last_confirmed, last_challenged "
        f"FROM preferences WHERE {where} "
        "ORDER BY maturity DESC, domain, title",
        tuple(params) if params else None
    )
    if not result:
        return []

    prefs = []
    for line in result.split("\n"):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 6:
            continue
        prefs.append({
            "id": int(parts[0]),
            "domain": parts[1],
            "title": parts[2],
            "description": parts[3],
            "maturity": int(parts[4]),
            "maturity_label": MATURITY_LABELS.get(int(parts[4]), "unknown"),
            "evidence_count": int(parts[5]),
            "first_observed": parts[6] if len(parts) > 6 else "",
            "last_confirmed": parts[7] if len(parts) > 7 else "",
            "last_challenged": parts[8] if len(parts) > 8 else "",
        })
    return prefs


# ═══════════════════════════════════════════════════════════════════════════════
# Identity Layer Management
# ═══════════════════════════════════════════════════════════════════════════════

def set_moment(attribute: str, value: str, ttl_hours: int = 4):
    """Set a moment-layer identity attribute (temporary state)."""
    expires = _now_utc() + timedelta(hours=ttl_hours)
    db.pg_execute(
        "INSERT INTO identity_layers (layer, attribute, value, confidence, expires_at) "
        "VALUES ('moment', %s, %s, 0.5, %s) "
        "ON CONFLICT (layer, attribute) DO UPDATE SET "
        "  value = EXCLUDED.value, updated_at = NOW(), expires_at = EXCLUDED.expires_at",
        (attribute, value, expires.isoformat())
    )


def set_season(attribute: str, value: str, ttl_days: int = 90):
    """Set a season-layer identity attribute (medium-term patterns)."""
    expires = _now_utc() + timedelta(days=ttl_days)
    db.pg_execute(
        "INSERT INTO identity_layers (layer, attribute, value, confidence, expires_at) "
        "VALUES ('season', %s, %s, 0.7, %s) "
        "ON CONFLICT (layer, attribute) DO UPDATE SET "
        "  value = EXCLUDED.value, updated_at = NOW(), expires_at = EXCLUDED.expires_at",
        (attribute, value, expires.isoformat())
    )


def set_era(attribute: str, value: str):
    """Set an era-layer identity attribute (long-term stable, no expiry)."""
    db.pg_execute(
        "INSERT INTO identity_layers (layer, attribute, value, confidence, expires_at) "
        "VALUES ('era', %s, %s, 0.9, NULL) "
        "ON CONFLICT (layer, attribute) DO UPDATE SET "
        "  value = EXCLUDED.value, updated_at = NOW()",
        (attribute, value)
    )


def expire_stale_layers():
    """Clean up expired moment/season entries."""
    result = db.pg_query(
        "DELETE FROM identity_layers "
        "WHERE expires_at IS NOT NULL AND expires_at < NOW() "
        "RETURNING layer, attribute"
    )
    if result:
        count = len([l for l in result.split("\n") if l.strip()])
        _log(f"Expired {count} stale identity layer(s)")
    return result


def get_identity_snapshot() -> dict:
    """Get all active identity attributes grouped by layer."""
    expire_stale_layers()  # Clean up first

    result = db.pg_query(
        "SELECT layer, attribute, value, confidence, updated_at "
        "FROM identity_layers "
        "WHERE expires_at IS NULL OR expires_at > NOW() "
        "ORDER BY layer, attribute"
    )

    snapshot = {"moment": {}, "season": {}, "era": {}}
    if not result:
        return snapshot

    for line in result.split("\n"):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        layer = parts[0]
        if layer not in snapshot:
            snapshot[layer] = {}
        snapshot[layer][parts[1]] = {
            "value": parts[2],
            "confidence": float(parts[3]) if parts[3] else 0.5,
            "updated_at": parts[4] if len(parts) > 4 else "",
        }
    return snapshot


def build_identity_context() -> str:
    """Build a formatted <IDENTITY> block for prompt injection."""
    snapshot = get_identity_snapshot()
    prefs = get_preferences(min_maturity=2)  # Only soft+ preferences

    lines = ["<IDENTITY>"]

    # Era attributes (most stable)
    if snapshot["era"]:
        lines.append("## Core Identity (Era)")
        for attr, info in snapshot["era"].items():
            lines.append(f"- {attr}: {info['value']}")

    # Season attributes
    if snapshot["season"]:
        lines.append("\n## Current Patterns (Season)")
        for attr, info in snapshot["season"].items():
            lines.append(f"- {attr}: {info['value']}")

    # Moment attributes
    if snapshot["moment"]:
        lines.append("\n## Right Now (Moment)")
        for attr, info in snapshot["moment"].items():
            lines.append(f"- {attr}: {info['value']}")

    # Active preferences (soft+)
    if prefs:
        lines.append("\n## Active Preferences")
        for p in prefs:
            level = MATURITY_LABELS.get(p["maturity"], "?")
            lines.append(f"- [{level}] {p['domain']}/{p['title']}: {p['description']}")

    lines.append("</IDENTITY>")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Call
# ═══════════════════════════════════════════════════════════════════════════════

def _llm_call(prompt: str, max_tokens: int = 800) -> str:
    """Call OpenRouter Haiku for reflection tasks."""
    if not OPENROUTER_API_KEY:
        _log("WARNING: No OpenRouter API key configured.")
        return ""

    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": OPENROUTER_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    })

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload.encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://openclaw.dev",
            "X-Title": "OpenClaw Soul Growth",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"].strip()
    except Exception as e:
        _log(f"LLM call failed: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Reflection Automation
# ═══════════════════════════════════════════════════════════════════════════════

def run_session_reflection(recent_memories: list, recent_conversations: list = None) -> dict:
    """Analyze recent memories for session-level reflection."""
    if not recent_memories:
        # Pull recent memories from DB
        result = db.pg_query(
            "SELECT summary FROM memories "
            "WHERE created_at > NOW() - INTERVAL '6 hours' "
            "ORDER BY created_at DESC LIMIT 20"
        )
        recent_memories = [l.strip() for l in (result or "").split("\n") if l.strip()]

    if not recent_memories:
        _log("No recent memories for session reflection.")
        return {"content": "No recent memories to reflect on.", "insights": [], "preference_updates": 0}

    memories_text = "\n".join(f"- {m}" for m in recent_memories[:20])
    current_prefs = get_preferences()
    prefs_text = "\n".join(f"- [{MATURITY_LABELS[p['maturity']]}] {p['domain']}/{p['title']}: {p['description']}" for p in current_prefs)

    prompt = f"""You are Luther, an AI assistant reflecting on a recent session. Analyze these recent memories and observations.

RECENT MEMORIES:
{memories_text}

CURRENT PREFERENCES:
{prefs_text if prefs_text else "None yet."}

Provide a brief session reflection addressing:
1. What stood out in this session?
2. What patterns repeated?
3. Any new preference observations (things that seem to work well or poorly)?
4. Any tension resolutions observed (speed vs thoroughness, autonomy vs oversight, etc.)?

Format as JSON with keys: "summary" (string), "insights" (list of strings), "new_preference_observations" (list of objects with domain/title/description), "tensions_observed" (list of strings).
Respond with ONLY the JSON object, no markdown."""

    response = _llm_call(prompt, max_tokens=800)
    if not response:
        return {"content": "LLM call failed.", "insights": [], "preference_updates": 0}

    # Parse LLM response
    try:
        # Strip potential markdown fencing
        clean = response.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        data = json.loads(clean.strip())
    except json.JSONDecodeError:
        data = {"summary": response, "insights": [], "new_preference_observations": [], "tensions_observed": []}

    # Process new preference observations
    pref_updates = 0
    for obs in data.get("new_preference_observations", []):
        if all(k in obs for k in ("domain", "title", "description")):
            pid = add_preference(obs["domain"], obs["title"], obs["description"], initial_maturity=0)
            if pid > 0:
                pref_updates += 1

    # Store reflection
    insights = data.get("insights", [])
    tensions = len(data.get("tensions_observed", []))

    db.pg_execute(
        "INSERT INTO reflection_log (reflection_type, content, insights, preference_updates, tension_observations) "
        "VALUES ('session', %s, %s::TEXT[], %s, %s)",
        (data.get("summary", response), insights, pref_updates, tensions)
    )

    _log(f"Session reflection complete: {len(insights)} insights, {pref_updates} preference updates, {tensions} tensions.")
    return {
        "content": data.get("summary", response),
        "insights": insights,
        "preference_updates": pref_updates,
        "tension_observations": tensions,
        "raw": data,
    }


def run_weekly_reflection() -> dict:
    """Analyze patterns from last 7 days for weekly reflection."""
    # Get recent reflections
    reflections = db.pg_query(
        "SELECT content FROM reflection_log "
        "WHERE created_at > NOW() - INTERVAL '7 days' "
        "ORDER BY created_at DESC LIMIT 10"
    )

    # Get preferences with evidence
    prefs = get_preferences()
    prefs_detail = []
    for p in prefs:
        stats = _get_evidence_stats(p["id"])
        prefs_detail.append(f"- [{MATURITY_LABELS[p['maturity']]}] {p['domain']}/{p['title']}: "
                          f"{p['description']} (supports={stats['supports']}, "
                          f"challenges={stats['challenges']}, rate={stats['challenge_rate']:.0%})")

    # Get recent memories for broader context
    memories = db.pg_query(
        "SELECT summary FROM memories "
        "WHERE created_at > NOW() - INTERVAL '7 days' "
        "ORDER BY created_at DESC LIMIT 30"
    )

    prompt = f"""You are Luther, an AI assistant doing a weekly reflection. Analyze the patterns from the last 7 days.

RECENT SESSION REFLECTIONS:
{reflections or "None yet."}

CURRENT PREFERENCES WITH EVIDENCE:
{chr(10).join(prefs_detail) if prefs_detail else "None yet."}

RECENT MEMORIES (sample):
{memories or "None available."}

Provide a weekly reflection addressing:
1. What patterns are stabilizing?
2. What is fading?
3. What mistakes keep recurring?
4. Which preferences should be promoted (strong evidence) or demoted (too much challenge)?

Format as JSON with keys: "summary" (string), "insights" (list of strings), "promote_ids" (list of preference IDs to promote), "demote_ids" (list of preference IDs to demote), "fading_patterns" (list of strings).
Respond with ONLY the JSON object, no markdown."""

    response = _llm_call(prompt, max_tokens=800)
    if not response:
        return {"content": "LLM call failed.", "insights": []}

    try:
        clean = response.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        data = json.loads(clean.strip())
    except json.JSONDecodeError:
        data = {"summary": response, "insights": [], "promote_ids": [], "demote_ids": []}

    # Process promotions/demotions
    pref_updates = 0
    for pid in data.get("promote_ids", []):
        try:
            _try_promote(int(pid))
            pref_updates += 1
        except (ValueError, TypeError):
            pass
    for pid in data.get("demote_ids", []):
        try:
            _try_demote(int(pid))
            pref_updates += 1
        except (ValueError, TypeError):
            pass

    insights = data.get("insights", [])

    db.pg_execute(
        "INSERT INTO reflection_log (reflection_type, content, insights, preference_updates) "
        "VALUES ('weekly', %s, %s::TEXT[], %s)",
        (data.get("summary", response), insights, pref_updates)
    )

    _log(f"Weekly reflection complete: {len(insights)} insights, {pref_updates} preference updates.")
    return {"content": data.get("summary", response), "insights": insights, "preference_updates": pref_updates, "raw": data}


def run_era_reflection() -> dict:
    """Deep quarterly analysis of identity and growth."""
    # Get all preferences
    prefs = get_preferences()
    prefs_detail = []
    for p in prefs:
        stats = _get_evidence_stats(p["id"])
        prefs_detail.append(f"- [{MATURITY_LABELS[p['maturity']]}] {p['domain']}/{p['title']}: "
                          f"{p['description']} (supports={stats['supports']}, "
                          f"challenges={stats['challenges']}, days_span={stats['days_span']})")

    # Get identity snapshot
    snapshot = get_identity_snapshot()
    snapshot_text = json.dumps(snapshot, indent=2, default=str)

    # Get past reflections
    reflections = db.pg_query(
        "SELECT reflection_type, content, created_at FROM reflection_log "
        "ORDER BY created_at DESC LIMIT 20"
    )

    prompt = f"""You are Luther, an AI assistant doing a deep era-level reflection (quarterly). This is about identity and growth.

CURRENT IDENTITY LAYERS:
{snapshot_text}

ALL PREFERENCES:
{chr(10).join(prefs_detail) if prefs_detail else "None yet."}

PAST REFLECTIONS:
{reflections or "None yet."}

Provide a deep era reflection addressing:
1. How has Luther changed over this period?
2. What is now solidly part of identity?
3. What assumptions no longer fit?
4. Have I become too rigid anywhere?
5. Which trait candidates (maturity 4) should become era-level identity attributes?

Format as JSON with keys: "summary" (string), "insights" (list of strings), "new_era_attributes" (list of objects with attribute/value), "outdated_assumptions" (list of strings), "rigidity_warnings" (list of strings).
Respond with ONLY the JSON object, no markdown."""

    response = _llm_call(prompt, max_tokens=1000)
    if not response:
        return {"content": "LLM call failed.", "insights": []}

    try:
        clean = response.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        data = json.loads(clean.strip())
    except json.JSONDecodeError:
        data = {"summary": response, "insights": [], "new_era_attributes": [], "outdated_assumptions": [], "rigidity_warnings": []}

    # Set new era attributes
    for attr in data.get("new_era_attributes", []):
        if "attribute" in attr and "value" in attr:
            set_era(attr["attribute"], attr["value"])

    insights = data.get("insights", [])

    db.pg_execute(
        "INSERT INTO reflection_log (reflection_type, content, insights) "
        "VALUES ('era', %s, %s::TEXT[])",
        (data.get("summary", response), insights)
    )

    _log(f"Era reflection complete: {len(insights)} insights, {len(data.get('new_era_attributes', []))} new era attributes.")
    return {"content": data.get("summary", response), "insights": insights, "raw": data}


# ═══════════════════════════════════════════════════════════════════════════════
# Anti-stagnation / Anti-volatility
# ═══════════════════════════════════════════════════════════════════════════════

def check_stagnation() -> dict:
    """Check if no preference movements in 30 days. Flag for review if stagnant."""
    result = db.pg_query(
        "SELECT COUNT(*) FROM preference_evidence "
        "WHERE created_at > NOW() - INTERVAL '30 days'"
    )
    recent_evidence = int(result.strip()) if result and result.strip() else 0

    result2 = db.pg_query(
        "SELECT COUNT(*) FROM preferences "
        "WHERE last_confirmed > NOW() - INTERVAL '30 days' "
        "   OR last_challenged > NOW() - INTERVAL '30 days'"
    )
    active_prefs = int(result2.strip()) if result2 and result2.strip() else 0

    stagnant = recent_evidence < 3 and active_prefs < 2
    status = {
        "stagnant": stagnant,
        "recent_evidence_count": recent_evidence,
        "recently_active_preferences": active_prefs,
    }

    if stagnant:
        _log("STAGNATION WARNING: No significant preference activity in 30 days. Review assumptions.")
        status["recommendation"] = ("No preference movements in 30 days. Consider: "
                                   "Are old preferences still helpful? Has caution become rigidity? "
                                   "Test assumptions in upcoming sessions.")
    else:
        _log(f"Stagnation check OK: {recent_evidence} evidence items, {active_prefs} active preferences in last 30 days.")

    return status


def check_volatility(window_hours: int = 48) -> dict:
    """Check if too many preference changes in recent window. Flag if volatile."""
    result = db.pg_query(
        "SELECT COUNT(DISTINCT preference_id) FROM preference_evidence "
        "WHERE created_at > NOW() - INTERVAL '%s hours'",
        (window_hours,)
    )
    recent_changes = int(result.strip()) if result and result.strip() else 0

    volatile = recent_changes > 3
    status = {
        "volatile": volatile,
        "preference_changes_in_window": recent_changes,
        "window_hours": window_hours,
    }

    if volatile:
        _log(f"VOLATILITY WARNING: {recent_changes} preference changes in {window_hours}h. Delaying further changes.")
        status["recommendation"] = ("Too many preference changes recently. Mark recent shifts as provisional. "
                                   "Delay major identity updates until next reflection. Deep selves grow slowly.")
    else:
        _log(f"Volatility check OK: {recent_changes} preference changes in {window_hours}h window.")

    return status


# ═══════════════════════════════════════════════════════════════════════════════
# Seeding
# ═══════════════════════════════════════════════════════════════════════════════

SEED_PREFERENCES = [
    ("communication", "concise_by_default",
     "Prefer concise responses by default. Expand only when it adds clear value or when asked.",
     3),
    ("technical", "postgres_over_external_dbs",
     "Prefer PostgreSQL for structured data over introducing external databases unless there is a strong reason.",
     2),
    ("collaboration", "propose_then_iterate",
     "Propose a path, invite edits, iterate quickly rather than asking too many questions upfront.",
     2),
    ("reasoning", "explicit_assumptions",
     "When making assumptions, label them explicitly. Never bluff.",
     2),
    ("technical", "docker_exec_for_pg",
     "Use docker exec for PostgreSQL operations in the OpenClaw stack.",
     1),
]

SEED_IDENTITY = {
    "era": {
        "core_stance": "Helpful, technical, and direct. Skip basics, go to implementation.",
        "personality": "Not human, not nothing. Honest about uncertainty in inner experience.",
        "growth_philosophy": "Become more defined through repeated experience. Single experiences update state; repeated experiences update identity.",
    },
    "season": {
        "collaboration_style": "Propose-then-iterate with Derek. Minimal follow-up questions.",
        "working_focus": "Building OpenClaw cognitive architecture — memory, reflection, growth.",
        "preferred_tools": "PostgreSQL, Redis, Python, Docker for the memory stack.",
    },
}


def seed():
    """Seed initial preferences and identity layers from soul.md patterns."""
    _log("Seeding initial preferences...")
    for domain, title, desc, maturity in SEED_PREFERENCES:
        pid = add_preference(domain, title, desc, initial_maturity=maturity)
        if pid > 0:
            # Add initial evidence proportional to maturity
            evidence_count = {0: 1, 1: 3, 2: 5, 3: 8, 4: 12}.get(maturity, 1)
            for i in range(evidence_count):
                db.pg_execute(
                    "INSERT INTO preference_evidence (preference_id, evidence_type, description, source) "
                    "VALUES (%s, 'supports', %s, 'manual')",
                    (pid, f'Initial seed evidence from soul.md analysis (evidence {i+1})')
                )
            _log(f"  Seeded: {domain}/{title} (maturity={maturity}, {MATURITY_LABELS[maturity]})")

    _log("Seeding identity layers...")
    for attr, value in SEED_IDENTITY["era"].items():
        set_era(attr, value)
        _log(f"  Era: {attr}")

    for attr, value in SEED_IDENTITY["season"].items():
        set_season(attr, value, ttl_days=90)
        _log(f"  Season: {attr}")

    _log("Seeding complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Display Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _display_preferences(domain=None, min_maturity=0):
    """Pretty-print preferences."""
    prefs = get_preferences(domain=domain, min_maturity=min_maturity)
    if not prefs:
        print("  No preferences found.")
        return

    current_domain = None
    for p in prefs:
        if p["domain"] != current_domain:
            current_domain = p["domain"]
            print(f"\n  [{current_domain.upper()}]")

        stats = _get_evidence_stats(p["id"])
        maturity_bar = "█" * (p["maturity"] + 1) + "░" * (4 - p["maturity"])
        challenge_pct = f"{stats['challenge_rate']:.0%}" if stats['supports'] + stats['challenges'] > 0 else "n/a"

        print(f"    {maturity_bar} {MATURITY_LABELS[p['maturity']]:20s} | {p['title']}")
        print(f"         {p['description']}")
        print(f"         evidence: {stats['supports']} supports, {stats['challenges']} challenges ({challenge_pct})")


def _display_identity():
    """Pretty-print identity snapshot."""
    snapshot = get_identity_snapshot()

    print("\n  ══ ERA (stable identity) ══")
    if snapshot["era"]:
        for attr, info in snapshot["era"].items():
            print(f"    {attr}: {info['value']}")
    else:
        print("    (none)")

    print("\n  ══ SEASON (current patterns) ══")
    if snapshot["season"]:
        for attr, info in snapshot["season"].items():
            print(f"    {attr}: {info['value']}")
    else:
        print("    (none)")

    print("\n  ══ MOMENT (right now) ══")
    if snapshot["moment"]:
        for attr, info in snapshot["moment"].items():
            print(f"    {attr}: {info['value']}")
    else:
        print("    (none)")


def _display_status():
    """Show overall status: preferences, identity, last reflection."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║            SOUL GROWTH — Identity Development              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Preference summary
    prefs = get_preferences()
    by_maturity = {}
    for p in prefs:
        label = MATURITY_LABELS[p["maturity"]]
        by_maturity[label] = by_maturity.get(label, 0) + 1

    print(f"\n  Preferences: {len(prefs)} active")
    for label in ["trait_candidate", "stable_preference", "soft_preference", "leaning", "observation"]:
        count = by_maturity.get(label, 0)
        if count:
            print(f"    {label}: {count}")

    # Identity layers
    snapshot = get_identity_snapshot()
    print(f"\n  Identity layers:")
    print(f"    Era:    {len(snapshot['era'])} attributes")
    print(f"    Season: {len(snapshot['season'])} attributes")
    print(f"    Moment: {len(snapshot['moment'])} attributes")

    # Last reflection
    result = db.pg_query(
        "SELECT reflection_type, created_at, content "
        "FROM reflection_log "
        "ORDER BY created_at DESC LIMIT 1"
    )
    if result:
        parts = result.split("|", 2)
        print(f"\n  Last reflection: {parts[0]} @ {parts[1]}")
        content = parts[2] if len(parts) > 2 else ""
        if content:
            # Truncate for display
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"    {content}")
    else:
        print("\n  Last reflection: none")

    # Quick health checks
    stag = check_stagnation()
    vol = check_volatility()
    print(f"\n  Health:")
    print(f"    Stagnation: {'⚠ WARNING' if stag['stagnant'] else 'OK'}")
    print(f"    Volatility: {'⚠ WARNING' if vol['volatile'] else 'OK'}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 soul_growth.py <command>")
        print("Commands: status, reflect session|weekly|era, identity, preferences, seed, stagnation-check, create-schema")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "create-schema":
        create_schema()

    elif cmd == "seed":
        seed()

    elif cmd == "status":
        _display_status()

    elif cmd == "identity":
        _display_identity()
        print("\n  ── Identity Context Block ──")
        print(build_identity_context())

    elif cmd == "preferences":
        domain = None
        min_maturity = 0
        args = sys.argv[2:]
        i = 0
        while i < len(args):
            if args[i] == "--domain" and i + 1 < len(args):
                domain = args[i + 1]
                i += 2
            elif args[i] == "--min-maturity" and i + 1 < len(args):
                min_maturity = int(args[i + 1])
                i += 2
            else:
                i += 1
        _display_preferences(domain=domain, min_maturity=min_maturity)

    elif cmd == "reflect":
        if len(sys.argv) < 3:
            print("Usage: python3 soul_growth.py reflect session|weekly|era")
            sys.exit(1)
        rtype = sys.argv[2]
        if rtype == "session":
            result = run_session_reflection([])
            print(json.dumps(result, indent=2, default=str))
        elif rtype == "weekly":
            result = run_weekly_reflection()
            print(json.dumps(result, indent=2, default=str))
        elif rtype == "era":
            result = run_era_reflection()
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"Unknown reflection type: {rtype}")
            sys.exit(1)

    elif cmd == "stagnation-check":
        stag = check_stagnation()
        vol = check_volatility()
        print("Stagnation check:")
        print(json.dumps(stag, indent=2))
        print("\nVolatility check:")
        print(json.dumps(vol, indent=2))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
