#!/usr/bin/env python3
"""
mental_models.py — Phase 7C: Mental Models + Prediction

Builds composite behavioral models from episodes, clusters, and relationships,
then uses them to anticipate what Derek might need next.

Three components:
  1. Entity Models — composite profiles for key entities (systems, projects, people)
     Aggregates: episode involvement, relationship density, topic affinity,
     failure patterns, lesson history
  2. Behavioral Patterns — time-of-day, topic switching, project momentum
  3. Predictive Context — "based on current context, likely next topics are X"

Minimum data thresholds — returns empty results if insufficient data.
Designed to activate ~48h after 7A/7B ship, once decay/clustering/episodes mature.

Schema:
  entity_models: entity_id, model_json, confidence, last_rebuilt
  behavioral_patterns: pattern_type, pattern_json, confidence, last_observed
  predictions: prediction_type, context_json, predicted_topics[], confidence, created_at

Cron: mental_models.py rebuild  (every 8h — rebuild models from fresh data)
      mental_models.py predict  (on-demand via daemon, not cron)

Usage:
  from mental_models import get_predictive_context, get_entity_model
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum data for meaningful models
MIN_EPISODES_FOR_MODELS = 15
MIN_CLUSTERS_FOR_PREDICTION = 5
MIN_ENTITY_MENTIONS = 3

# How many top entities to build models for
MAX_ENTITY_MODELS = 50

# Prediction context limit
PREDICTION_CONTEXT_LIMIT = 3


def _log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [mental_models] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════════════════════

def migrate():
    """Create mental model tables. Idempotent."""
    conn = db.get_pg()
    if not conn:
        _log("ERROR: No PG connection")
        return False

    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS entity_models (
            id SERIAL PRIMARY KEY,
            entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
            entity_name TEXT NOT NULL,
            model JSONB NOT NULL DEFAULT '{}',
            confidence REAL DEFAULT 0.0,
            episode_count INTEGER DEFAULT 0,
            relationship_count INTEGER DEFAULT 0,
            last_rebuilt TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (entity_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS behavioral_patterns (
            id SERIAL PRIMARY KEY,
            pattern_type TEXT NOT NULL,
            pattern JSONB NOT NULL DEFAULT '{}',
            confidence REAL DEFAULT 0.0,
            observation_count INTEGER DEFAULT 0,
            last_observed TIMESTAMPTZ DEFAULT NOW(),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (pattern_type)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions_log (
            id SERIAL PRIMARY KEY,
            query TEXT,
            predicted_topics TEXT[] DEFAULT '{}',
            predicted_entities TEXT[] DEFAULT '{}',
            confidence REAL DEFAULT 0.0,
            context JSONB DEFAULT '{}',
            was_useful BOOLEAN DEFAULT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_entity_models_name ON entity_models(entity_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions_log(created_at DESC)")

    conn.commit()
    _log("Migration complete")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Data readiness check
# ═══════════════════════════════════════════════════════════════════════════════

def check_data_readiness() -> dict:
    """Check if we have enough data for meaningful models."""
    conn = db.get_pg()
    if not conn:
        return {"ready": False, "reason": "no_pg"}

    cur = conn.cursor()

    cur.execute("SELECT count(*) FROM episodes")
    episodes = cur.fetchone()[0]

    cur.execute("SELECT count(DISTINCT cluster_id) FROM entity_clusters")
    clusters = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM relationships WHERE observation_count > 1")
    reinforced = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM causal_chains WHERE lesson != '' AND lesson IS NOT NULL")
    lessons = cur.fetchone()[0]

    ready = (
        episodes >= MIN_EPISODES_FOR_MODELS
        and clusters >= MIN_CLUSTERS_FOR_PREDICTION
    )

    return {
        "ready": ready,
        "episodes": episodes,
        "clusters": clusters,
        "reinforced_edges": reinforced,
        "lessons": lessons,
        "thresholds": {
            "min_episodes": MIN_EPISODES_FOR_MODELS,
            "min_clusters": MIN_CLUSTERS_FOR_PREDICTION,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Entity Models
# ═══════════════════════════════════════════════════════════════════════════════

def _build_entity_model(entity_name: str, entity_id: int, cur) -> dict:
    """Build a composite model for one entity."""

    model = {
        "name": entity_name,
        "episode_involvement": [],
        "topics": [],
        "co_occurring_entities": [],
        "relationship_summary": [],
        "failure_patterns": [],
        "lessons": [],
        "cluster": None,
    }

    # Episodes involving this entity
    cur.execute("""
        SELECT title, outcome, topics, started_at
        FROM episodes
        WHERE %s = ANY(entities)
        ORDER BY started_at DESC
        LIMIT 10
    """, (entity_name,))
    for title, outcome, topics, started in cur.fetchall():
        model["episode_involvement"].append({
            "title": title,
            "outcome": outcome,
            "when": str(started) if started else None,
        })
        if topics:
            model["topics"].extend(topics)

    # Topic frequency
    topic_counts = Counter(model["topics"])
    model["topics"] = [{"topic": t, "count": c} for t, c in topic_counts.most_common(5)]

    # Co-occurring entities (from same episodes)
    cur.execute("""
        SELECT e, count(*) as cnt
        FROM (
            SELECT unnest(entities) as e
            FROM episodes
            WHERE %s = ANY(entities)
        ) sub
        WHERE e != %s
        GROUP BY e
        ORDER BY cnt DESC
        LIMIT 8
    """, (entity_name, entity_name))
    model["co_occurring_entities"] = [
        {"entity": r[0], "co_occurrences": r[1]} for r in cur.fetchall()
    ]

    # Relationship summary
    cur.execute("""
        SELECT e2.name, r.predicate, r.observation_count, r.decay_score
        FROM relationships r
        JOIN entities e2 ON e2.id = r.target_id
        WHERE r.source_id = %s AND r.decay_score > 0.1
        ORDER BY r.observation_count DESC, r.decay_score DESC
        LIMIT 8
    """, (entity_id,))
    for name, pred, obs, decay in cur.fetchall():
        model["relationship_summary"].append({
            "target": name, "predicate": pred,
            "strength": round(obs * decay, 2),
        })

    # Also inbound relationships
    cur.execute("""
        SELECT e2.name, r.predicate, r.observation_count, r.decay_score
        FROM relationships r
        JOIN entities e2 ON e2.id = r.source_id
        WHERE r.target_id = %s AND r.decay_score > 0.1
        ORDER BY r.observation_count DESC, r.decay_score DESC
        LIMIT 8
    """, (entity_id,))
    for name, pred, obs, decay in cur.fetchall():
        model["relationship_summary"].append({
            "source": name, "predicate": pred,
            "strength": round(obs * decay, 2),
        })

    # Lessons involving this entity
    cur.execute("""
        SELECT c.lesson, c.trigger, e.title
        FROM causal_chains c
        JOIN episodes e ON e.id = c.episode_id
        WHERE %s = ANY(c.entities) AND c.lesson != '' AND c.lesson IS NOT NULL
        ORDER BY e.ended_at DESC
        LIMIT 5
    """, (entity_name,))
    model["lessons"] = [
        {"lesson": r[0], "trigger": r[1], "episode": r[2]}
        for r in cur.fetchall()
    ]

    # Failure patterns (episodes with non-completed outcomes)
    cur.execute("""
        SELECT title, outcome
        FROM episodes
        WHERE %s = ANY(entities)
          AND outcome NOT IN ('completed', 'clarified')
        ORDER BY started_at DESC
        LIMIT 5
    """, (entity_name,))
    model["failure_patterns"] = [
        {"title": r[0], "outcome": r[1]} for r in cur.fetchall()
    ]

    # Cluster membership
    cur.execute("""
        SELECT ec.cluster_id, ec.cluster_label
        FROM entity_clusters ec
        WHERE ec.entity_id = %s
    """, (entity_id,))
    row = cur.fetchone()
    if row:
        model["cluster"] = {"id": row[0], "label": row[1]}

    # Calculate confidence based on data density
    episode_ct = len(model["episode_involvement"])
    rel_ct = len(model["relationship_summary"])
    confidence = min(1.0, (episode_ct * 0.1 + rel_ct * 0.05 + len(model["lessons"]) * 0.1))

    return model, confidence, episode_ct, rel_ct


def rebuild_entity_models() -> int:
    """Rebuild entity models for top entities."""
    conn = db.get_pg()
    if not conn:
        return 0

    cur = conn.cursor()

    # Find entities that appear in episodes most frequently
    cur.execute("""
        SELECT unnest(entities) as e, count(*) as cnt
        FROM episodes
        GROUP BY e
        HAVING count(*) >= %s
        ORDER BY cnt DESC
        LIMIT %s
    """, (MIN_ENTITY_MENTIONS, MAX_ENTITY_MODELS))

    top_entities = cur.fetchall()
    if not top_entities:
        _log("No entities meet minimum mention threshold")
        return 0

    built = 0
    for entity_name, mention_count in top_entities:
        # Resolve entity ID
        cur.execute(
            "SELECT id FROM entities WHERE name ILIKE %s LIMIT 1",
            (entity_name,)
        )
        row = cur.fetchone()
        if not row:
            continue
        entity_id = row[0]

        model, confidence, ep_ct, rel_ct = _build_entity_model(
            entity_name, entity_id, cur
        )

        cur.execute("""
            INSERT INTO entity_models
                (entity_id, entity_name, model, confidence, episode_count, relationship_count)
            VALUES (%s, %s, %s::jsonb, %s, %s, %s)
            ON CONFLICT (entity_id) DO UPDATE SET
                model = EXCLUDED.model,
                confidence = EXCLUDED.confidence,
                episode_count = EXCLUDED.episode_count,
                relationship_count = EXCLUDED.relationship_count,
                last_rebuilt = NOW()
        """, (entity_id, entity_name, json.dumps(model), confidence, ep_ct, rel_ct))

        built += 1

    conn.commit()
    _log(f"Built {built} entity models from {len(top_entities)} candidates")
    return built


# ═══════════════════════════════════════════════════════════════════════════════
# Behavioral Patterns
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_behavioral_patterns() -> int:
    """Detect and store behavioral patterns from episodes."""
    conn = db.get_pg()
    if not conn:
        return 0

    cur = conn.cursor()
    patterns_found = 0

    # 1. Topic affinity — which topics cluster together?
    cur.execute("""
        SELECT topics FROM episodes WHERE array_length(topics, 1) > 0
    """)
    topic_pairs = Counter()
    topic_freq = Counter()
    for (topics,) in cur.fetchall():
        if not topics:
            continue
        for t in topics:
            topic_freq[t] += 1
        for i, t1 in enumerate(topics):
            for t2 in topics[i + 1:]:
                pair = tuple(sorted([t1, t2]))
                topic_pairs[pair] += 1

    if topic_freq:
        pattern = {
            "top_topics": [{"topic": t, "count": c} for t, c in topic_freq.most_common(10)],
            "topic_pairs": [
                {"pair": list(p), "co_occurrence": c}
                for p, c in topic_pairs.most_common(10) if c >= 2
            ],
        }
        _store_pattern(cur, "topic_affinity", pattern, len(topic_freq))
        patterns_found += 1

    # 2. Project momentum — which topics have recent activity vs. going stale?
    cur.execute("""
        SELECT unnest(topics) as t,
               MAX(started_at) as last_active,
               count(*) as episode_count,
               count(*) FILTER (WHERE outcome = 'completed') as completed
        FROM episodes
        WHERE started_at IS NOT NULL
        GROUP BY t
        ORDER BY last_active DESC
    """)
    momentum = []
    now = datetime.now(timezone.utc)
    for topic, last_active, ep_count, completed in cur.fetchall():
        if last_active:
            days_since = (now - last_active).days
            completion_rate = completed / max(ep_count, 1)
            momentum.append({
                "topic": topic,
                "last_active_days_ago": days_since,
                "episode_count": ep_count,
                "completion_rate": round(completion_rate, 2),
                "status": "active" if days_since < 3 else "cooling" if days_since < 7 else "stale",
            })

    if momentum:
        _store_pattern(cur, "project_momentum", {"topics": momentum}, len(momentum))
        patterns_found += 1

    # 3. Outcome patterns — what tends to succeed vs. stay incomplete?
    cur.execute("""
        SELECT unnest(entities) as e, outcome, count(*)
        FROM episodes
        GROUP BY e, outcome
        ORDER BY e, count DESC
    """)
    entity_outcomes = defaultdict(lambda: Counter())
    for entity, outcome, count in cur.fetchall():
        entity_outcomes[entity][outcome] += count

    # Find entities with high failure rates
    trouble_entities = []
    for entity, outcomes in entity_outcomes.items():
        total = sum(outcomes.values())
        if total < 2:
            continue
        completed = outcomes.get("completed", 0)
        failure_rate = 1 - (completed / total)
        if failure_rate > 0.4:
            trouble_entities.append({
                "entity": entity,
                "total_episodes": total,
                "completion_rate": round(1 - failure_rate, 2),
                "outcomes": dict(outcomes),
            })

    if trouble_entities:
        trouble_entities.sort(key=lambda x: x["completion_rate"])
        _store_pattern(cur, "trouble_spots", {"entities": trouble_entities[:10]}, len(trouble_entities))
        patterns_found += 1

    # 4. Session complexity — average turns per topic
    cur.execute("""
        SELECT unnest(topics) as t, AVG(turn_count) as avg_turns, MAX(turn_count) as max_turns
        FROM episodes
        WHERE turn_count > 0
        GROUP BY t
        HAVING count(*) >= 2
        ORDER BY AVG(turn_count) DESC
    """)
    complexity = [
        {"topic": r[0], "avg_turns": round(float(r[1]), 1), "max_turns": int(r[2])}
        for r in cur.fetchall()
    ]
    if complexity:
        _store_pattern(cur, "session_complexity", {"topics": complexity}, len(complexity))
        patterns_found += 1

    # 5. Lesson density — which topics generate the most lessons?
    cur.execute("""
        SELECT unnest(e.topics) as t, count(c.id) as lesson_count
        FROM episodes e
        JOIN causal_chains c ON c.episode_id = e.id
        WHERE c.lesson != '' AND c.lesson IS NOT NULL
        GROUP BY t
        ORDER BY lesson_count DESC
        LIMIT 10
    """)
    lesson_density = [{"topic": r[0], "lessons": r[1]} for r in cur.fetchall()]
    if lesson_density:
        _store_pattern(cur, "lesson_density", {"topics": lesson_density}, len(lesson_density))
        patterns_found += 1

    conn.commit()
    _log(f"Analyzed {patterns_found} behavioral patterns")
    return patterns_found


def _store_pattern(cur, pattern_type: str, pattern: dict, observation_count: int):
    """Upsert a behavioral pattern."""
    confidence = min(1.0, observation_count * 0.05)
    cur.execute("""
        INSERT INTO behavioral_patterns
            (pattern_type, pattern, confidence, observation_count, last_observed)
        VALUES (%s, %s::jsonb, %s, %s, NOW())
        ON CONFLICT (pattern_type) DO UPDATE SET
            pattern = EXCLUDED.pattern,
            confidence = EXCLUDED.confidence,
            observation_count = EXCLUDED.observation_count,
            last_observed = NOW()
    """, (pattern_type, json.dumps(pattern), confidence, observation_count))


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction
# ═══════════════════════════════════════════════════════════════════════════════

def predict_context(query: str, entity_names: list[str] = None) -> dict:
    """
    Given a query and optional entity names, predict:
    - Likely related topics
    - Entities that might be relevant
    - Recent lessons that apply
    - Related episodes

    Returns a prediction dict with confidence scores.
    """
    conn = db.get_pg()
    if not conn:
        return {"predictions": [], "confidence": 0}

    cur = conn.cursor()

    readiness = check_data_readiness()
    if not readiness["ready"]:
        return {"predictions": [], "confidence": 0, "reason": "insufficient_data",
                "readiness": readiness}

    predictions = {}

    # 1. Entity model lookups — if query mentions known entities, pull their models
    related_entities = []
    if entity_names:
        for name in entity_names[:5]:
            cur.execute("""
                SELECT model, confidence FROM entity_models
                WHERE entity_name ILIKE %s
            """, (name,))
            row = cur.fetchone()
            if row:
                model = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                related_entities.append({
                    "entity": name,
                    "co_occurs_with": [
                        e["entity"] for e in model.get("co_occurring_entities", [])[:5]
                    ],
                    "topics": [t["topic"] for t in model.get("topics", [])[:3]],
                    "recent_lessons": [
                        l["lesson"] for l in model.get("lessons", [])[:2]
                    ],
                    "cluster": model.get("cluster"),
                    "confidence": row[1],
                })

    predictions["related_entities"] = related_entities

    # 2. Topic prediction — based on entity models and recent momentum
    predicted_topics = Counter()
    for ent in related_entities:
        for t in ent.get("topics", []):
            predicted_topics[t] += 1

    # Boost active topics from momentum
    cur.execute("""
        SELECT pattern FROM behavioral_patterns
        WHERE pattern_type = 'project_momentum'
    """)
    row = cur.fetchone()
    if row:
        momentum = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        for topic_info in momentum.get("topics", []):
            if topic_info.get("status") == "active":
                predicted_topics[topic_info["topic"]] += 0.5

    predictions["predicted_topics"] = [
        {"topic": t, "score": round(s, 2)}
        for t, s in predicted_topics.most_common(5)
    ]

    # 3. Applicable lessons — from causal chains matching query entities
    applicable_lessons = []
    if entity_names:
        cur.execute("""
            SELECT c.lesson, c.trigger, e.title
            FROM causal_chains c
            JOIN episodes e ON e.id = c.episode_id
            WHERE c.entities && %s
              AND c.lesson != '' AND c.lesson IS NOT NULL
            ORDER BY e.ended_at DESC
            LIMIT 3
        """, (list(entity_names),))
        applicable_lessons = [
            {"lesson": r[0], "trigger": r[1], "from_episode": r[2]}
            for r in cur.fetchall()
        ]

    predictions["applicable_lessons"] = applicable_lessons

    # 4. Trouble spots — warn if query involves entities with high failure rates
    trouble_warnings = []
    cur.execute("""
        SELECT pattern FROM behavioral_patterns
        WHERE pattern_type = 'trouble_spots'
    """)
    row = cur.fetchone()
    if row and entity_names:
        trouble = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        trouble_names = {e["entity"].lower() for e in trouble.get("entities", [])}
        for name in entity_names:
            if name.lower() in trouble_names:
                for e in trouble.get("entities", []):
                    if e["entity"].lower() == name.lower():
                        trouble_warnings.append({
                            "entity": name,
                            "completion_rate": e["completion_rate"],
                            "total_episodes": e["total_episodes"],
                        })

    predictions["trouble_warnings"] = trouble_warnings

    # Overall confidence
    data_points = (
        len(related_entities) +
        len(predicted_topics) +
        len(applicable_lessons)
    )
    confidence = min(0.9, data_points * 0.1)
    predictions["confidence"] = round(confidence, 2)

    return predictions


def build_predictive_context(query: str, entity_names: list[str] = None) -> str:
    """
    Build a <PREDICTIVE_CONTEXT> block for prompt injection.
    Only produces output if predictions are meaningful.
    """
    pred = predict_context(query, entity_names)
    if pred.get("confidence", 0) < 0.2:
        return ""

    lines = ["<PREDICTIVE_CONTEXT>"]

    # Applicable lessons
    if pred.get("applicable_lessons"):
        lines.append("Relevant lessons from past work:")
        for l in pred["applicable_lessons"]:
            lines.append(f"  - {l['lesson']}")
            if l.get("trigger"):
                lines.append(f"    (triggered by: {l['trigger']})")

    # Trouble warnings
    if pred.get("trouble_warnings"):
        lines.append("⚠ Entities with historically low completion rates:")
        for w in pred["trouble_warnings"]:
            lines.append(
                f"  - {w['entity']}: {w['completion_rate']*100:.0f}% completion "
                f"across {w['total_episodes']} episodes"
            )

    # Related entities (from entity models)
    if pred.get("related_entities"):
        entities_context = []
        for ent in pred["related_entities"]:
            if ent.get("co_occurs_with"):
                entities_context.append(
                    f"  {ent['entity']} often involves: {', '.join(ent['co_occurs_with'][:3])}"
                )
        if entities_context:
            lines.append("Related context:")
            lines.extend(entities_context)

    if len(lines) <= 1:
        return ""

    lines.append("</PREDICTIVE_CONTEXT>")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Entity model retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def get_entity_model(entity_name: str) -> dict | None:
    """Get the composite model for an entity."""
    conn = db.get_pg()
    if not conn:
        return None

    cur = conn.cursor()
    cur.execute("""
        SELECT model, confidence, episode_count, relationship_count, last_rebuilt
        FROM entity_models WHERE entity_name ILIKE %s
    """, (entity_name,))
    row = cur.fetchone()
    if not row:
        return None

    model = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    model["_meta"] = {
        "confidence": row[1],
        "episode_count": row[2],
        "relationship_count": row[3],
        "last_rebuilt": str(row[4]),
    }
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Full rebuild pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def rebuild_all():
    """Full rebuild: check readiness → entity models → behavioral patterns."""
    readiness = check_data_readiness()
    _log(f"Data readiness: {json.dumps(readiness)}")

    if not readiness["ready"]:
        _log("Insufficient data for model building. Skipping.")
        return

    models = rebuild_entity_models()
    patterns = analyze_behavioral_patterns()
    _log(f"Rebuild complete: {models} entity models, {patterns} behavioral patterns")


# ═══════════════════════════════════════════════════════════════════════════════
# Daemon handlers
# ═══════════════════════════════════════════════════════════════════════════════

def handle_predict(params: dict) -> dict:
    """Predict relevant context for a query."""
    query = params.get("query", "")
    entities = params.get("entities", [])
    if isinstance(entities, str):
        entities = [entities]

    pred = predict_context(query, entity_names=entities or None)
    return {"ok": True, **pred}


def handle_predictive_context(params: dict) -> dict:
    """Build predictive context block for injection."""
    query = params.get("query", "")
    entities = params.get("entities", [])
    if isinstance(entities, str):
        entities = [entities]

    ctx = build_predictive_context(query, entity_names=entities or None)
    return {"ok": True, "context": ctx}


def handle_entity_model(params: dict) -> dict:
    """Get entity model."""
    name = params.get("entity", "")
    if not name:
        return {"ok": False, "error": "Missing 'entity'"}

    model = get_entity_model(name)
    if not model:
        return {"ok": True, "model": None, "message": f"No model for '{name}'"}

    return {"ok": True, "model": model}


def handle_model_stats(params: dict) -> dict:
    """Return mental model statistics."""
    conn = db.get_pg()
    if not conn:
        return {"ok": False, "error": "No PG connection"}

    cur = conn.cursor()

    readiness = check_data_readiness()

    cur.execute("SELECT count(*) FROM entity_models")
    models = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM behavioral_patterns")
    patterns = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM predictions_log")
    predictions = cur.fetchone()[0]

    # Top entity models by confidence
    cur.execute("""
        SELECT entity_name, confidence, episode_count
        FROM entity_models
        ORDER BY confidence DESC
        LIMIT 10
    """)
    top_models = [
        {"entity": r[0], "confidence": r[1], "episodes": r[2]}
        for r in cur.fetchall()
    ]

    # Pattern types
    cur.execute("""
        SELECT pattern_type, confidence, observation_count, last_observed
        FROM behavioral_patterns
        ORDER BY last_observed DESC
    """)
    pattern_info = [
        {"type": r[0], "confidence": r[1], "observations": r[2],
         "last_observed": str(r[3])}
        for r in cur.fetchall()
    ]

    return {
        "ok": True,
        "readiness": readiness,
        "entity_models": models,
        "behavioral_patterns": patterns,
        "predictions_logged": predictions,
        "top_models": top_models,
        "patterns": pattern_info,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Stats (CLI)
# ═══════════════════════════════════════════════════════════════════════════════

def show_stats():
    """Print mental model statistics."""
    conn = db.get_pg()
    if not conn:
        print("ERROR: No PG connection")
        return

    cur = conn.cursor()

    readiness = check_data_readiness()
    print(f"\n  Mental Models — Phase 7C")
    print(f"  {'='*50}")
    print(f"  Data ready:    {'YES' if readiness['ready'] else 'NO'}")
    print(f"  Episodes:      {readiness['episodes']} (need {readiness['thresholds']['min_episodes']})")
    print(f"  Clusters:      {readiness['clusters']} (need {readiness['thresholds']['min_clusters']})")
    print(f"  Reinforced:    {readiness['reinforced_edges']} edges with obs > 1")
    print(f"  Lessons:       {readiness['lessons']}")

    cur.execute("SELECT count(*) FROM entity_models")
    models = cur.fetchone()[0]
    print(f"\n  Entity models: {models}")

    if models > 0:
        cur.execute("""
            SELECT entity_name, confidence, episode_count, relationship_count
            FROM entity_models
            ORDER BY confidence DESC
            LIMIT 15
        """)
        print(f"  Top models:")
        for name, conf, eps, rels in cur.fetchall():
            print(f"    {name:30s} conf={conf:.2f} eps={eps} rels={rels}")

    cur.execute("SELECT count(*) FROM behavioral_patterns")
    patterns = cur.fetchone()[0]
    print(f"\n  Behavioral patterns: {patterns}")

    if patterns > 0:
        cur.execute("""
            SELECT pattern_type, confidence, observation_count
            FROM behavioral_patterns
            ORDER BY last_observed DESC
        """)
        for ptype, conf, obs in cur.fetchall():
            print(f"    {ptype:25s} conf={conf:.2f} obs={obs}")

    # Show momentum if available
    cur.execute("""
        SELECT pattern FROM behavioral_patterns
        WHERE pattern_type = 'project_momentum'
    """)
    row = cur.fetchone()
    if row:
        momentum = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        active = [t for t in momentum.get("topics", []) if t.get("status") == "active"]
        cooling = [t for t in momentum.get("topics", []) if t.get("status") == "cooling"]
        stale = [t for t in momentum.get("topics", []) if t.get("status") == "stale"]
        if active or cooling:
            print(f"\n  Project momentum:")
            for t in active:
                print(f"    🟢 {t['topic']} ({t['episode_count']} eps, {t['completion_rate']*100:.0f}% done)")
            for t in cooling:
                print(f"    🟡 {t['topic']} ({t['episode_count']} eps, last {t['last_active_days_ago']}d ago)")
            for t in stale[:3]:
                print(f"    🔴 {t['topic']} ({t['episode_count']} eps, last {t['last_active_days_ago']}d ago)")

    # Show trouble spots if available
    cur.execute("""
        SELECT pattern FROM behavioral_patterns
        WHERE pattern_type = 'trouble_spots'
    """)
    row = cur.fetchone()
    if row:
        trouble = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        entities = trouble.get("entities", [])[:5]
        if entities:
            print(f"\n  Trouble spots (low completion rate):")
            for e in entities:
                print(f"    ⚠ {e['entity']:30s} {e['completion_rate']*100:.0f}% completion ({e['total_episodes']} eps)")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mental Models — Phase 7C")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("migrate", help="Apply schema migrations")
    sub.add_parser("stats", help="Show model stats")
    sub.add_parser("rebuild", help="Full rebuild: entity models + behavioral patterns")
    sub.add_parser("readiness", help="Check data readiness")

    p_predict = sub.add_parser("predict", help="Predict context for query")
    p_predict.add_argument("query", help="Query text")
    p_predict.add_argument("--entities", nargs="*", help="Entity names")

    p_model = sub.add_parser("model", help="Show entity model")
    p_model.add_argument("entity", help="Entity name")

    p_ctx = sub.add_parser("context", help="Build predictive context")
    p_ctx.add_argument("query", help="Query text")
    p_ctx.add_argument("--entities", nargs="*", help="Entity names")

    args = parser.parse_args()

    if args.command == "migrate":
        migrate()

    elif args.command == "stats":
        show_stats()

    elif args.command == "rebuild":
        migrate()
        rebuild_all()
        show_stats()

    elif args.command == "readiness":
        r = check_data_readiness()
        print(json.dumps(r, indent=2))

    elif args.command == "predict":
        pred = predict_context(args.query, entity_names=args.entities)
        print(json.dumps(pred, indent=2, default=str))

    elif args.command == "model":
        model = get_entity_model(args.entity)
        if model:
            print(json.dumps(model, indent=2, default=str))
        else:
            print(f"No model for '{args.entity}'")

    elif args.command == "context":
        ctx = build_predictive_context(args.query, entity_names=args.entities)
        print(ctx if ctx else "No predictive context (insufficient data or confidence)")

    else:
        parser.print_help()
