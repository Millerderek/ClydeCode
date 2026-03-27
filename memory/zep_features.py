#!/usr/bin/env python3
"""
zep_features.py — Three features adapted from Zep's design for OpenClaw.

1. Fact Invalidation Chain — immutable temporal audit trail for facts
2. Ontology System — typed entity/edge schemas with constraints
3. Pluggable Rerankers — RRF, MMR, and diversity-aware reranking

Run: python3 zep_features.py migrate     # Apply schema changes
     python3 zep_features.py seed        # Seed ontology types
     python3 zep_features.py status      # Show current state
"""

import json
import os
import sys
import math
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FACT INVALIDATION CHAIN
# ═══════════════════════════════════════════════════════════════════════════════

FACT_MIGRATION_SQL = """
-- Temporal columns for immutable fact chain
ALTER TABLE facts ADD COLUMN IF NOT EXISTS valid_from
    TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE facts ADD COLUMN IF NOT EXISTS invalidated_at
    TIMESTAMPTZ DEFAULT NULL;
ALTER TABLE facts ADD COLUMN IF NOT EXISTS invalidated_by
    INTEGER DEFAULT NULL REFERENCES facts(id);
ALTER TABLE facts ADD COLUMN IF NOT EXISTS invalidation_reason
    TEXT DEFAULT NULL;

-- Index for efficient "current facts" queries
CREATE INDEX IF NOT EXISTS idx_facts_valid
    ON facts (topic_id) WHERE invalidated_at IS NULL;

-- Index for audit trail lookups
CREATE INDEX IF NOT EXISTS idx_facts_invalidated_by
    ON facts (invalidated_by) WHERE invalidated_by IS NOT NULL;

-- Backfill: set valid_from = created_at for existing facts
UPDATE facts SET valid_from = created_at WHERE valid_from IS NULL;
"""


def invalidate_fact(fact_id: int, new_fact_id: int = None, reason: str = "superseded") -> bool:
    """
    Invalidate a fact without deleting it. Creates an audit trail.

    Args:
        fact_id: the fact to invalidate
        new_fact_id: optional — the fact that supersedes this one
        reason: 'superseded', 'contradiction', 'stale', 'manual', 'compaction'

    Returns:
        True if the fact was invalidated
    """
    valid_reasons = {"superseded", "contradiction", "stale", "manual", "compaction"}
    if reason not in valid_reasons:
        reason = "manual"

    if new_fact_id:
        db.pg_execute(
            "UPDATE facts SET invalidated_at = NOW(), invalidated_by = %s, "
            "invalidation_reason = %s WHERE id = %s AND invalidated_at IS NULL;",
            (new_fact_id, reason, fact_id)
        )
    else:
        db.pg_execute(
            "UPDATE facts SET invalidated_at = NOW(), "
            "invalidation_reason = %s WHERE id = %s AND invalidated_at IS NULL;",
            (reason, fact_id)
        )
    return True


def add_fact_with_invalidation(topic_id: int, content: str, memory_id=None,
                                source: str = "auto") -> int:
    """
    Add a new fact and automatically invalidate any contradicting facts
    in the same topic that share significant word overlap.

    Returns the new fact's ID.
    """
    import hashlib
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    # Check for duplicate
    existing = db.pg_query(
        "SELECT id FROM facts WHERE content_hash = %s AND invalidated_at IS NULL LIMIT 1;",
        (content_hash,)
    )
    if existing:
        try:
            return int(existing.strip())
        except ValueError:
            pass

    # Insert new fact
    if memory_id:
        result = db.pg_query(
            "INSERT INTO facts (topic_id, memory_id, content, content_hash, source, valid_from) "
            "VALUES (%s, %s, %s, %s, %s, NOW()) RETURNING id;",
            (topic_id, memory_id, content, content_hash, source)
        )
    else:
        result = db.pg_query(
            "INSERT INTO facts (topic_id, content, content_hash, source, valid_from) "
            "VALUES (%s, %s, %s, %s, NOW()) RETURNING id;",
            (topic_id, content, content_hash, source)
        )

    if not result:
        return 0
    new_id = int(result.strip())

    # Auto-invalidate contradicting facts in same topic (simple heuristic)
    # Look for facts with high word overlap but different content
    if topic_id:
        _auto_invalidate_contradictions(topic_id, new_id, content)

    return new_id


def _auto_invalidate_contradictions(topic_id: int, new_fact_id: int, new_content: str):
    """
    Check for facts in the same topic that appear to be superseded by the new fact.
    Uses word overlap heuristic — if >60% of significant words match but content differs,
    the old fact is likely superseded.
    """
    import re
    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "was", "one", "our", "has", "how", "its", "may", "new", "now", "see",
        "way", "who", "did", "get", "use", "this", "that", "with", "have",
        "from", "they", "been", "some", "what", "when", "will", "more", "into",
        "also", "than", "them", "very", "just", "about", "which", "their",
    }
    new_words = set(
        w.lower() for w in re.findall(r'[a-zA-Z]{3,}', new_content)
        if w.lower() not in stopwords
    )
    if len(new_words) < 3:
        return

    # Get active facts in same topic
    rows = db.pg_query(
        "SELECT id, content FROM facts WHERE topic_id = %s AND id != %s "
        "AND invalidated_at IS NULL;",
        (topic_id, new_fact_id)
    )
    if not rows:
        return

    for line in rows.split("\n"):
        if "|" not in line:
            continue
        parts = line.split("|", 1)
        if len(parts) < 2:
            continue
        try:
            old_id = int(parts[0])
        except ValueError:
            continue
        old_content = parts[1]

        old_words = set(
            w.lower() for w in re.findall(r'[a-zA-Z]{3,}', old_content)
            if w.lower() not in stopwords
        )
        if len(old_words) < 3:
            continue

        overlap = new_words & old_words
        overlap_ratio = len(overlap) / min(len(new_words), len(old_words))

        # High overlap (>60%) but different content = likely superseded
        if overlap_ratio > 0.6 and old_content.strip() != new_content.strip():
            invalidate_fact(old_id, new_fact_id, "superseded")
            print(f"  [fact-chain] Invalidated fact {old_id} (superseded by {new_fact_id})")


def get_current_facts(topic_id: int = None, include_history: bool = False) -> list:
    """
    Get facts, optionally filtered by topic.
    By default returns only current (non-invalidated) facts.
    With include_history=True, returns all facts with their temporal status.
    """
    if include_history:
        where = "WHERE topic_id = %s" if topic_id else ""
        params = (topic_id,) if topic_id else ()
        result = db.pg_query(
            f"SELECT id, topic_id, content, valid_from, invalidated_at, "
            f"invalidation_reason, invalidated_by "
            f"FROM facts {where} ORDER BY valid_from DESC;",
            params
        )
    else:
        where = "WHERE invalidated_at IS NULL" + (" AND topic_id = %s" if topic_id else "")
        params = (topic_id,) if topic_id else ()
        result = db.pg_query(
            f"SELECT id, topic_id, content, valid_from "
            f"FROM facts {where} ORDER BY valid_from DESC;",
            params
        )

    if not result:
        return []

    facts = []
    for line in result.split("\n"):
        if "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) >= 4:
            fact = {
                "id": int(parts[0]),
                "topic_id": int(parts[1]) if parts[1].strip() else None,
                "content": parts[2],
                "valid_from": parts[3],
            }
            if include_history and len(parts) >= 7:
                fact["invalidated_at"] = parts[4] if parts[4].strip() else None
                fact["invalidation_reason"] = parts[5] if parts[5].strip() else None
                fact["invalidated_by"] = int(parts[6]) if parts[6].strip() else None
            facts.append(fact)
    return facts


def get_fact_chain(fact_id: int) -> list:
    """
    Get the full chain of superseding facts for a given fact.
    Returns chronological list: [oldest_version, ..., current_version]
    """
    # Walk backwards to find the root
    chain = []
    visited = set()
    current = fact_id

    # First walk forward from invalidated_by to find predecessors
    while current:
        if current in visited:
            break
        visited.add(current)

        result = db.pg_query(
            "SELECT id, content, valid_from, invalidated_at, invalidated_by "
            "FROM facts WHERE invalidated_by = %s;",
            (current,)
        )
        if not result or "|" not in result:
            break
        parts = result.split("|")
        predecessor = int(parts[0])
        chain.insert(0, {
            "id": predecessor,
            "content": parts[1],
            "valid_from": parts[2],
            "invalidated_at": parts[3],
            "status": "invalidated",
        })
        current = predecessor

    # Now add the queried fact and walk forward through superseding facts
    current = fact_id
    visited_forward = set()
    while current:
        if current in visited_forward:
            break
        visited_forward.add(current)

        result = db.pg_query(
            "SELECT id, content, valid_from, invalidated_at, invalidated_by "
            "FROM facts WHERE id = %s;",
            (current,)
        )
        if not result or "|" not in result:
            break
        parts = result.split("|")
        is_invalid = bool(parts[3].strip())
        chain.append({
            "id": int(parts[0]),
            "content": parts[1],
            "valid_from": parts[2],
            "invalidated_at": parts[3] if is_invalid else None,
            "status": "invalidated" if is_invalid else "current",
        })

        # Find what supersedes this fact
        successor = db.pg_query(
            "SELECT id FROM facts WHERE invalidated_by IS NOT NULL "
            "AND id = (SELECT invalidated_by FROM facts WHERE id = %s);",
            (current,)
        )
        if not successor:
            # Try the other direction — find the fact that has invalidated_by = current
            successor = db.pg_query(
                "SELECT id FROM facts WHERE id IN "
                "(SELECT id FROM facts WHERE invalidated_at IS NOT NULL) "
                "LIMIT 1;"  # This needs rethinking
            )
            break
        current = int(successor.strip())

    return chain


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ONTOLOGY SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

ONTOLOGY_MIGRATION_SQL = """
-- Entity type definitions (ontology)
CREATE TABLE IF NOT EXISTS entity_types (
    name        TEXT PRIMARY KEY,
    parent_type TEXT REFERENCES entity_types(name),
    description TEXT,
    properties  JSONB DEFAULT '{}',  -- schema for expected properties
    icon        TEXT DEFAULT NULL,    -- optional display hint
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Edge type definitions (ontology)
CREATE TABLE IF NOT EXISTS edge_types (
    name                TEXT PRIMARY KEY,
    description         TEXT,
    source_types        TEXT[] DEFAULT '{}',  -- allowed source entity types
    target_types        TEXT[] DEFAULT '{}',  -- allowed target entity types
    properties          JSONB DEFAULT '{}',   -- schema for expected properties
    is_directional      BOOLEAN DEFAULT TRUE,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Add ontology FK to entities (soft — allows values not in ontology for backwards compat)
-- We don't add a hard FK since existing data may have types not yet in ontology
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities (type);

-- Add description column to entities if missing
ALTER TABLE entities ADD COLUMN IF NOT EXISTS description TEXT DEFAULT NULL;
"""

# Default ontology seed — based on actual types in use + domain-specific additions
ENTITY_TYPE_SEED = [
    # Infrastructure
    ("infrastructure", None, "Physical or virtual infrastructure components",
     '{"expected_props": ["ip", "provider", "location"]}'),
    ("device", "infrastructure", "Physical hardware devices",
     '{"expected_props": ["ip", "model", "firmware"]}'),
    ("service", None, "Software services, APIs, daemons",
     '{"expected_props": ["port", "protocol", "endpoint"]}'),
    ("system", None, "Software systems, platforms, applications",
     '{"expected_props": ["version", "vendor"]}'),

    # Code & Automation
    ("script", None, "Scripts, automation code, config files",
     '{"expected_props": ["language", "path", "cron_schedule"]}'),
    ("agent", None, "AI agents, bots, automated actors",
     '{"expected_props": ["model", "runtime", "version"]}'),

    # People & Organizations
    ("person", None, "Human individuals",
     '{"expected_props": ["role", "organization"]}'),
    ("client", None, "Business clients, customers, accounts",
     '{"expected_props": ["industry", "contract_type"]}'),

    # Communication & Network
    ("protocol", None, "Communication protocols (MQTT, SIP, Modbus, etc.)",
     '{"expected_props": ["port", "version"]}'),

    # Domain-specific (ANJ inverter, home automation)
    ("register", "device", "Hardware registers (Modbus, I2C, etc.)",
     '{"expected_props": ["address", "access", "data_type", "unit"]}'),
    ("sensor", "device", "Sensors and measurement devices",
     '{"expected_props": ["unit", "mqtt_topic"]}'),
    ("automation", "script", "Home automation rules, HA automations",
     '{"expected_props": ["trigger_type", "platform"]}'),
    ("container", "service", "Docker containers",
     '{"expected_props": ["image", "network_mode", "restart_policy"]}'),
    ("topic", None, "MQTT topics, message channels",
     '{"expected_props": ["qos", "retained"]}'),
    ("credential", None, "API keys, passwords, auth tokens",
     '{"expected_props": ["service", "expires"]}'),

    # Catch-all
    ("general", None, "Unclassified entities", '{}'),
]

EDGE_TYPE_SEED = [
    ("USES", "Entity uses/consumes another", '{}', True,
     [], []),
    ("CONFIGURED_BY", "Entity is configured by another", '{}', True,
     [], ["script", "person", "agent"]),
    ("DEPENDS_ON", "Entity depends on another to function", '{}', True,
     [], []),
    ("RUNS_ON", "Entity runs on infrastructure/platform", '{}', True,
     ["service", "script", "agent", "container"], ["infrastructure", "device", "system"]),
    ("CONNECTED_TO", "Entity has network/data connection to another", '{}', False,
     [], []),
    ("PART_OF", "Entity is a component of another", '{}', True,
     [], []),
    ("MONITORS", "Entity monitors/observes another", '{}', True,
     ["agent", "script", "service"], []),
    ("BACKS_UP", "Entity backs up another", '{}', True,
     ["script", "service"], []),
    ("SERVES", "Entity serves/provides to another", '{}', True,
     ["service", "system"], []),
    ("AUTHENTICATES_WITH", "Entity authenticates via another", '{}', True,
     [], ["credential", "service"]),
    ("MIGRATING_FROM", "Entity is being migrated from another", '{}', True,
     [], []),
    ("MANAGED_BY", "Entity is managed/administered by another", '{}', True,
     [], ["person", "agent", "script"]),
    ("REPORTS_TO", "Entity reports to another", '{}', True,
     ["person", "agent"], ["person"]),
    ("INVOLVES", "Entity is involved with another", '{}', False,
     [], []),
    ("READS_FROM", "Entity reads data from another", '{}', True,
     [], ["register", "sensor", "topic"]),
    ("WRITES_TO", "Entity writes data to another", '{}', True,
     [], ["register", "topic"]),
    ("PUBLISHES_TO", "Entity publishes messages to another", '{}', True,
     ["service", "script", "agent", "container"], ["topic"]),
    ("SUBSCRIBES_TO", "Entity subscribes to messages from another", '{}', True,
     ["service", "script", "agent", "container"], ["topic"]),
]


def seed_ontology():
    """Seed entity and edge type definitions."""
    seeded_entities = 0
    seeded_edges = 0

    for name, parent, desc, props in ENTITY_TYPE_SEED:
        db.pg_execute(
            "INSERT INTO entity_types (name, parent_type, description, properties) "
            "VALUES (%s, %s, %s, %s::jsonb) "
            "ON CONFLICT (name) DO UPDATE SET "
            "description = EXCLUDED.description, properties = EXCLUDED.properties;",
            (name, parent, desc, props)
        )
        seeded_entities += 1

    for name, desc, props, directional, source_types, target_types in EDGE_TYPE_SEED:
        db.pg_execute(
            "INSERT INTO edge_types (name, description, properties, is_directional, "
            "source_types, target_types) "
            "VALUES (%s, %s, %s::jsonb, %s, %s, %s) "
            "ON CONFLICT (name) DO UPDATE SET "
            "description = EXCLUDED.description, source_types = EXCLUDED.source_types, "
            "target_types = EXCLUDED.target_types;",
            (name, desc, props, directional, source_types, target_types)
        )
        seeded_edges += 1

    print(f"  Seeded {seeded_entities} entity types, {seeded_edges} edge types")


def validate_triple(subject_type: str, predicate: str, object_type: str) -> dict:
    """
    Validate a triple against the ontology.
    Returns: {"valid": bool, "warnings": [...]}
    """
    warnings = []

    # Check entity types exist
    for etype, label in [(subject_type, "subject"), (object_type, "object")]:
        result = db.pg_query(
            "SELECT name FROM entity_types WHERE name = %s;", (etype,)
        )
        if not result:
            warnings.append(f"{label} type '{etype}' not in ontology")

    # Check edge type exists and constraints
    result = db.pg_query(
        "SELECT source_types, target_types FROM edge_types WHERE name = %s;",
        (predicate,)
    )
    if not result:
        warnings.append(f"predicate '{predicate}' not in ontology")
    elif "|" in result:
        parts = result.split("|")
        source_types = parts[0].strip("{}").split(",") if parts[0].strip("{}") else []
        target_types = parts[1].strip("{}").split(",") if parts[1].strip("{}") else []

        if source_types and subject_type not in source_types:
            warnings.append(
                f"predicate '{predicate}' expects source types {source_types}, "
                f"got '{subject_type}'"
            )
        if target_types and object_type not in target_types:
            warnings.append(
                f"predicate '{predicate}' expects target types {target_types}, "
                f"got '{object_type}'"
            )

    return {"valid": len(warnings) == 0, "warnings": warnings}


def get_entity_type_hierarchy() -> dict:
    """Get the full entity type hierarchy as a nested dict."""
    result = db.pg_query(
        "SELECT name, parent_type, description FROM entity_types ORDER BY name;"
    )
    if not result:
        return {}

    types = {}
    for line in result.split("\n"):
        if "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) >= 3:
            types[parts[0]] = {
                "parent": parts[1] if parts[1].strip() else None,
                "description": parts[2],
            }
    return types


def reclassify_general_entities(dry_run: bool = True) -> list:
    """
    Suggest reclassifications for 'general' type entities based on their
    relationships and connected entity types.

    Returns list of (entity_id, entity_name, suggested_type, reason)
    """
    # Get general entities with their relationships
    result = db.pg_query(
        "SELECT e.id, e.name, r.predicate, "
        "CASE WHEN r.source_id = e.id THEN e2.type ELSE e2.type END as connected_type, "
        "CASE WHEN r.source_id = e.id THEN 'outbound' ELSE 'inbound' END as direction "
        "FROM entities e "
        "JOIN relationships r ON (r.source_id = e.id OR r.target_id = e.id) "
        "JOIN entities e2 ON (e2.id = CASE WHEN r.source_id = e.id THEN r.target_id ELSE r.source_id END) "
        "WHERE e.type = 'general' "
        "ORDER BY e.id;"
    )
    if not result:
        return []

    # Group by entity
    entity_signals = defaultdict(list)
    entity_names = {}
    for line in result.split("\n"):
        if "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) >= 5:
            eid = int(parts[0])
            entity_names[eid] = parts[1]
            entity_signals[eid].append({
                "predicate": parts[2],
                "connected_type": parts[3],
                "direction": parts[4],
            })

    suggestions = []
    for eid, signals in entity_signals.items():
        name = entity_names[eid]

        # Heuristic rules for reclassification
        predicates = [s["predicate"] for s in signals]
        connected_types = [s["connected_type"] for s in signals]

        suggested = None
        reason = ""

        if "RUNS_ON" in predicates and any(d == "outbound" for s in signals
                                            if s["predicate"] == "RUNS_ON"
                                            for d in [s["direction"]]):
            suggested = "service"
            reason = "has RUNS_ON relationship (outbound)"
        elif "CONFIGURED_BY" in predicates:
            suggested = "system"
            reason = "has CONFIGURED_BY relationship"
        elif "MONITORS" in predicates and any(s["direction"] == "outbound"
                                               for s in signals if s["predicate"] == "MONITORS"):
            suggested = "agent"
            reason = "monitors other entities"
        elif "PART_OF" in predicates and "device" in connected_types:
            suggested = "device"
            reason = "is PART_OF a device"
        elif "PART_OF" in predicates and "service" in connected_types:
            suggested = "service"
            reason = "is PART_OF a service"

        if suggested:
            suggestions.append((eid, name, suggested, reason))
            if not dry_run:
                db.pg_execute(
                    "UPDATE entities SET type = %s, updated_at = NOW() WHERE id = %s;",
                    (suggested, eid)
                )

    return suggestions


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PLUGGABLE RERANKERS
# ═══════════════════════════════════════════════════════════════════════════════

def reciprocal_rank_fusion(ranked_lists: list, k: int = 60) -> list:
    """
    Reciprocal Rank Fusion (RRF) — combines multiple ranked result lists.

    Each result gets a score of 1/(k + rank) from each list it appears in.
    Final score is the sum across all lists.

    Args:
        ranked_lists: list of lists, each containing dicts with 'id' and 'memory' keys
        k: smoothing constant (default 60, standard in literature)

    Returns:
        Fused list sorted by RRF score, each dict gains 'rrf_score' and 'rrf_ranks' keys
    """
    scores = defaultdict(float)
    items = {}
    rank_info = defaultdict(list)

    for list_idx, ranked_list in enumerate(ranked_lists):
        for rank, item in enumerate(ranked_list):
            item_id = item.get("id", item.get("mem_id", ""))
            if not item_id:
                continue
            rrf_contribution = 1.0 / (k + rank + 1)
            scores[item_id] += rrf_contribution
            items[item_id] = item
            rank_info[item_id].append((list_idx, rank + 1))

    # Build fused result
    fused = []
    for item_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        item = dict(items[item_id])
        item["rrf_score"] = round(score, 6)
        item["rrf_ranks"] = rank_info[item_id]
        fused.append(item)

    return fused


def maximal_marginal_relevance(candidates: list, query_embedding=None,
                                 lambda_param: float = 0.7,
                                 limit: int = 5) -> list:
    """
    Maximal Marginal Relevance (MMR) — diversity-aware reranking.

    Balances relevance with diversity by penalizing candidates that are
    too similar to already-selected results.

    Since we don't always have embeddings, this uses text-based similarity
    (Jaccard on word sets) as a proxy for embedding cosine similarity.

    Args:
        candidates: list of dicts with 'memory' and 'score'/'final' keys
        query_embedding: unused (for API compat) — we use text similarity
        lambda_param: balance between relevance (1.0) and diversity (0.0)
                     0.7 = mostly relevant with some diversity push
        limit: max results to return

    Returns:
        Reranked list with 'mmr_score' added to each dict
    """
    import re

    if not candidates:
        return []

    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "was", "one", "our", "has", "how", "its", "may", "new", "now", "see",
        "way", "who", "did", "get", "use", "this", "that", "with", "have",
        "from", "they", "been", "some", "what", "when", "will", "more", "into",
        "also", "than", "them", "very", "just", "about", "which", "their",
    }

    def _word_set(text):
        return set(
            w.lower() for w in re.findall(r'[a-zA-Z]{3,}', text or "")
            if w.lower() not in stopwords
        )

    def _jaccard(set_a, set_b):
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    # Pre-compute word sets
    word_sets = [_word_set(c.get("memory", "")) for c in candidates]
    relevance_scores = [c.get("final", c.get("score", 0.0)) for c in candidates]

    selected = []
    selected_indices = set()
    remaining = set(range(len(candidates)))

    for _ in range(min(limit, len(candidates))):
        best_idx = None
        best_mmr = -float("inf")

        for idx in remaining:
            relevance = relevance_scores[idx]

            # Max similarity to any already-selected item
            if selected_indices:
                max_sim = max(
                    _jaccard(word_sets[idx], word_sets[s_idx])
                    for s_idx in selected_indices
                )
            else:
                max_sim = 0.0

            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx

        if best_idx is not None:
            item = dict(candidates[best_idx])
            item["mmr_score"] = round(best_mmr, 6)
            selected.append(item)
            selected_indices.add(best_idx)
            remaining.discard(best_idx)

    return selected


def rerank(candidates: list, strategy: str = "salience",
           limit: int = 5, **kwargs) -> list:
    """
    Pluggable reranker interface.

    Args:
        candidates: list of search result dicts
        strategy: one of 'salience', 'rrf', 'mmr', 'rrf+mmr'
        limit: max results
        **kwargs: strategy-specific params (lambda_param, k, ranked_lists)

    Returns:
        Reranked list
    """
    if strategy == "salience":
        # Default — use existing salience_engine
        try:
            from salience_engine import salience_score
            query = kwargs.get("query", "")
            weights = kwargs.get("weights", None)
            return salience_score(query, candidates, limit=limit, weights=weights)
        except ImportError:
            return candidates[:limit]

    elif strategy == "rrf":
        # RRF requires multiple ranked lists
        ranked_lists = kwargs.get("ranked_lists", [candidates])
        k = kwargs.get("k", 60)
        return reciprocal_rank_fusion(ranked_lists, k=k)[:limit]

    elif strategy == "mmr":
        lambda_param = kwargs.get("lambda_param", 0.7)
        return maximal_marginal_relevance(
            candidates, lambda_param=lambda_param, limit=limit
        )

    elif strategy == "rrf+mmr":
        # First fuse with RRF, then diversify with MMR
        ranked_lists = kwargs.get("ranked_lists", [candidates])
        k = kwargs.get("k", 60)
        fused = reciprocal_rank_fusion(ranked_lists, k=k)

        lambda_param = kwargs.get("lambda_param", 0.7)
        return maximal_marginal_relevance(
            fused, lambda_param=lambda_param, limit=limit
        )

    elif strategy == "salience+mmr":
        # Salience score first, then MMR for diversity
        try:
            from salience_engine import salience_score
            query = kwargs.get("query", "")
            weights = kwargs.get("weights", None)
            scored = salience_score(query, candidates, limit=limit * 2, weights=weights)
        except ImportError:
            scored = candidates[:limit * 2]

        lambda_param = kwargs.get("lambda_param", 0.7)
        return maximal_marginal_relevance(
            scored, lambda_param=lambda_param, limit=limit
        )

    else:
        print(f"  [rerank] Unknown strategy: {strategy}, falling back to truncation")
        return candidates[:limit]


# ═══════════════════════════════════════════════════════════════════════════════
# Migration runner
# ═══════════════════════════════════════════════════════════════════════════════

def migrate():
    """Apply all schema migrations."""
    print("  Applying fact invalidation chain...")
    for stmt in FACT_MIGRATION_SQL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            try:
                db.pg_execute(stmt + ";")
            except Exception as e:
                if "already exists" not in str(e) and "duplicate" not in str(e).lower():
                    print(f"    Warning: {e}")

    print("  Applying ontology tables...")
    for stmt in ONTOLOGY_MIGRATION_SQL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            try:
                db.pg_execute(stmt + ";")
            except Exception as e:
                if "already exists" not in str(e) and "duplicate" not in str(e).lower():
                    print(f"    Warning: {e}")

    print("  Migrations applied.")


def status():
    """Show current state of all three features."""
    print("\n  === Fact Invalidation Chain ===")
    total = db.pg_query("SELECT COUNT(*) FROM facts;")
    active = db.pg_query("SELECT COUNT(*) FROM facts WHERE invalidated_at IS NULL;")
    invalidated = db.pg_query(
        "SELECT COUNT(*) FROM facts WHERE invalidated_at IS NOT NULL;"
    )
    print(f"  Total facts: {total}")
    print(f"  Active: {active}")
    print(f"  Invalidated: {invalidated}")

    # Check if columns exist
    col_check = db.pg_query(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'facts' AND column_name = 'invalidated_at';"
    )
    print(f"  Schema migrated: {'yes' if col_check else 'no'}")

    print("\n  === Ontology System ===")
    et_count = db.pg_query("SELECT COUNT(*) FROM entity_types;")
    edge_count = db.pg_query("SELECT COUNT(*) FROM edge_types;")
    general_count = db.pg_query(
        "SELECT COUNT(*) FROM entities WHERE type = 'general';"
    )
    print(f"  Entity types defined: {et_count}")
    print(f"  Edge types defined: {edge_count}")
    print(f"  Entities with type='general': {general_count} (candidates for reclassification)")

    # Type distribution
    print("\n  Entity type distribution:")
    dist = db.pg_query(
        "SELECT type, COUNT(*) FROM entities GROUP BY type ORDER BY COUNT(*) DESC LIMIT 10;"
    )
    if dist:
        for line in dist.split("\n"):
            if "|" in line:
                parts = line.split("|")
                print(f"    {parts[0]:20s} {parts[1]}")

    print("\n  === Reranker Strategies ===")
    strategies = ["salience", "rrf", "mmr", "rrf+mmr", "salience+mmr"]
    print(f"  Available: {', '.join(strategies)}")
    print(f"  Default: salience (existing behavior)")
    print(f"  Recommended: salience+mmr (adds diversity)")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Zep-inspired features for OpenClaw")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("migrate", help="Apply schema migrations")
    sub.add_parser("seed", help="Seed ontology types")
    sub.add_parser("status", help="Show current state")

    p_chain = sub.add_parser("chain", help="Show fact chain for a fact ID")
    p_chain.add_argument("fact_id", type=int)

    p_facts = sub.add_parser("facts", help="List current facts")
    p_facts.add_argument("--topic", type=int, default=None)
    p_facts.add_argument("--history", action="store_true")
    p_facts.add_argument("--limit", type=int, default=20)

    p_ontology = sub.add_parser("ontology", help="Show ontology hierarchy")
    p_validate = sub.add_parser("validate", help="Validate a triple")
    p_validate.add_argument("subject_type")
    p_validate.add_argument("predicate")
    p_validate.add_argument("object_type")

    p_reclass = sub.add_parser("reclassify", help="Suggest reclassifications for general entities")
    p_reclass.add_argument("--apply", action="store_true", help="Apply suggestions")

    p_rerank = sub.add_parser("rerank-test", help="Test reranker strategies on a query")
    p_rerank.add_argument("query")
    p_rerank.add_argument("--strategy", default="salience+mmr")
    p_rerank.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    DIM = "\033[2m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    if args.command == "migrate":
        migrate()

    elif args.command == "seed":
        seed_ontology()

    elif args.command == "status":
        status()

    elif args.command == "chain":
        chain = get_fact_chain(args.fact_id)
        if not chain:
            print("  No chain found")
        else:
            for i, f in enumerate(chain):
                marker = "→" if f["status"] == "current" else "✗"
                inv = f" (invalidated: {f.get('invalidated_at', '')})" if f["status"] == "invalidated" else ""
                print(f"  {marker} [{f['id']}] {f['content'][:80]}{inv}")

    elif args.command == "facts":
        facts = get_current_facts(topic_id=args.topic, include_history=args.history)
        for f in facts[:args.limit]:
            status_mark = GREEN + "●" + RESET if not f.get("invalidated_at") else RED + "✗" + RESET
            print(f"  {status_mark} [{f['id']}] {f['content'][:90]}")
            if f.get("invalidated_at"):
                print(f"    {DIM}invalidated: {f['invalidated_at']} ({f.get('invalidation_reason', '?')}){RESET}")

    elif args.command == "ontology":
        hierarchy = get_entity_type_hierarchy()
        # Print as tree
        roots = [n for n, t in hierarchy.items() if t["parent"] is None]
        def _print_tree(name, depth=0):
            info = hierarchy.get(name, {})
            indent = "  " * depth + ("├── " if depth > 0 else "")
            print(f"  {indent}{CYAN}{name}{RESET} — {info.get('description', '')}")
            children = [n for n, t in hierarchy.items() if t["parent"] == name]
            for child in sorted(children):
                _print_tree(child, depth + 1)
        for root in sorted(roots):
            _print_tree(root)

    elif args.command == "validate":
        result = validate_triple(args.subject_type, args.predicate, args.object_type)
        if result["valid"]:
            print(f"  {GREEN}✓ Valid triple{RESET}")
        else:
            print(f"  {YELLOW}⚠ Warnings:{RESET}")
            for w in result["warnings"]:
                print(f"    - {w}")

    elif args.command == "reclassify":
        suggestions = reclassify_general_entities(dry_run=not args.apply)
        if not suggestions:
            print("  No reclassification suggestions")
        else:
            action = "Applied" if args.apply else "Suggested"
            print(f"  {action} {len(suggestions)} reclassifications:")
            for eid, name, suggested, reason in suggestions[:20]:
                print(f"    [{eid}] {name}: general → {CYAN}{suggested}{RESET} ({reason})")
            if len(suggestions) > 20:
                print(f"    ... and {len(suggestions) - 20} more")

    elif args.command == "rerank-test":
        # Fetch candidates via Mem0 and test reranker
        try:
            from openclaw_memo import get_memory, CLYDE_USER
            m = get_memory()
            raw = m.search(args.query, user_id=CLYDE_USER, limit=15)
            candidates = raw.get("results", [])
        except Exception as e:
            print(f"  {RED}Cannot fetch candidates: {e}{RESET}")
            sys.exit(1)

        if not candidates:
            print("  No candidates found")
            sys.exit(0)

        results = rerank(candidates, strategy=args.strategy,
                        limit=args.limit, query=args.query)
        print(f"\n  {BOLD}Reranked with '{args.strategy}' ({len(results)} results):{RESET}\n")
        for i, r in enumerate(results, 1):
            score_key = "mmr_score" if "mmr_score" in r else "rrf_score" if "rrf_score" in r else "final"
            score = r.get(score_key, r.get("score", 0))
            mem = r.get("memory", "")[:70]
            print(f"  {i:2d}. [{score:.4f}] {mem}")

    else:
        parser.print_help()
