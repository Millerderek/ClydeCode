#!/usr/bin/env python3
"""
graph_evolution.py — Phase 7A: Relationship Evolution + Graph Clustering

Turns the static entity graph into a living system:
  1. Temporal tracking: observation_count, last_confirmed, decay_score on edges
  2. Decay scoring: edges weaken over time if not re-observed
  3. Community detection: label propagation clusters entities into functional groups
  4. Cluster-aware retrieval: given an entity, return its full cluster context

Schema additions (applied via migrate()):
  - relationships: +observation_count, +last_confirmed, +decay_score
  - entity_clusters: cluster_id, entity_id, cluster_label, confidence, updated_at

Cron: graph_evolution.py decay    (every 6h — recompute decay scores)
       graph_evolution.py cluster  (every 12h — recompute clusters)
       graph_evolution.py stats    (show current state)

Can also be imported:
  from graph_evolution import get_cluster_context, decay_relationships
"""

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

# Decay: half-life in days — after this many days without re-observation,
# decay_score drops to 0.5. Double the half-life → 0.25, etc.
DECAY_HALF_LIFE_DAYS = float(os.environ.get("CLYDE_DECAY_HALF_LIFE", "14"))

# Clustering: minimum edges for an entity to participate in clustering
MIN_EDGES_FOR_CLUSTER = 2

# Clustering: max iterations for label propagation
CLUSTER_MAX_ITERATIONS = 50

# Clustering: minimum cluster size to keep
MIN_CLUSTER_SIZE = 3


def _log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [graph_evolution] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Schema migration
# ═══════════════════════════════════════════════════════════════════════════════

def migrate():
    """Add evolution columns + cluster table. Idempotent."""
    conn = db.get_pg()
    if not conn:
        _log("ERROR: No PG connection")
        return False

    cur = conn.cursor()

    # Add columns to relationships (idempotent via IF NOT EXISTS pattern)
    migrations = [
        "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS observation_count INTEGER DEFAULT 1",
        "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS last_confirmed TIMESTAMPTZ DEFAULT NOW()",
        "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS decay_score REAL DEFAULT 1.0",
    ]

    for sql in migrations:
        try:
            cur.execute(sql)
        except Exception as e:
            _log(f"Migration warning: {e}")

    # Create entity_clusters table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS entity_clusters (
            id SERIAL PRIMARY KEY,
            cluster_id INTEGER NOT NULL,
            entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            cluster_label TEXT DEFAULT '',
            confidence REAL DEFAULT 1.0,
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (entity_id)
        )
    """)

    # Index for fast cluster lookups
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_entity_clusters_cluster_id
        ON entity_clusters (cluster_id)
    """)

    # Cluster metadata table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cluster_meta (
            cluster_id INTEGER PRIMARY KEY,
            label TEXT DEFAULT '',
            entity_count INTEGER DEFAULT 0,
            top_entities TEXT[] DEFAULT '{}',
            top_predicates TEXT[] DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    conn.commit()
    _log("Migration complete")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Relationship evolution — called from graph_extractor on upsert
# ═══════════════════════════════════════════════════════════════════════════════

def reinforce_relationship(source_id: int, target_id: int, predicate: str,
                           props: dict = None):
    """
    Upsert a relationship with evolution tracking.
    If the edge already exists: increment observation_count, refresh last_confirmed,
    reset decay_score to 1.0.
    If new: insert with observation_count=1.

    Drop-in replacement for graph_extractor._upsert_relationship.
    """
    props_json = json.dumps(props or {})

    db.pg_execute(
        """INSERT INTO relationships
           (source_id, target_id, predicate, properties,
            observation_count, last_confirmed, decay_score)
           VALUES (%s, %s, %s, %s::jsonb, 1, NOW(), 1.0)
           ON CONFLICT (source_id, target_id, predicate) DO UPDATE SET
             properties = %s::jsonb,
             observation_count = relationships.observation_count + 1,
             last_confirmed = NOW(),
             decay_score = 1.0
        """,
        (source_id, target_id, predicate, props_json, props_json)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Decay scoring
# ═══════════════════════════════════════════════════════════════════════════════

def decay_relationships():
    """
    Recompute decay_score for all relationships based on time since last_confirmed.

    Formula: decay_score = 2^(-(days_since_confirmed / half_life))
    - At 0 days: 1.0
    - At half_life days: 0.5
    - At 2x half_life: 0.25
    - etc.

    Observation count provides a floor: heavily observed relationships decay slower.
    Adjusted formula: decay = 2^(-(days / (half_life * log2(obs_count + 1))))
    This means a relationship observed 8 times has 3x the half-life.
    """
    conn = db.get_pg()
    if not conn:
        _log("ERROR: No PG connection")
        return 0

    cur = conn.cursor()

    # Compute decay for all relationships in one UPDATE
    # Using ln(2) = 0.693... and log base conversion
    cur.execute("""
        UPDATE relationships SET
            decay_score = POWER(
                2.0,
                -(EXTRACT(EPOCH FROM (NOW() - COALESCE(last_confirmed, created_at))) / 86400.0)
                / (%s * (LN(GREATEST(observation_count, 1)::FLOAT + 1.0) / LN(2.0)))
            )
        WHERE last_confirmed IS NOT NULL OR created_at IS NOT NULL
        RETURNING id
    """, (DECAY_HALF_LIFE_DAYS,))

    updated = len(cur.fetchall())
    conn.commit()

    # Stats
    cur.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE decay_score > 0.8) as strong,
            COUNT(*) FILTER (WHERE decay_score BETWEEN 0.4 AND 0.8) as moderate,
            COUNT(*) FILTER (WHERE decay_score BETWEEN 0.1 AND 0.4) as weak,
            COUNT(*) FILTER (WHERE decay_score < 0.1) as dead
        FROM relationships
    """)
    total, strong, moderate, weak, dead = cur.fetchone()

    _log(f"Decay update: {updated} relationships — "
         f"strong={strong}, moderate={moderate}, weak={weak}, dead={dead}")

    return updated


# ═══════════════════════════════════════════════════════════════════════════════
# Community detection — Label Propagation
# ═══════════════════════════════════════════════════════════════════════════════

def _build_adjacency() -> tuple[dict, dict]:
    """
    Build adjacency list from relationships table.
    Returns (adjacency_dict, entity_info_dict).

    Only includes entities with >= MIN_EDGES_FOR_CLUSTER connections
    and relationships with decay_score > 0.1 (skip dead edges).
    """
    conn = db.get_pg()
    if not conn:
        return {}, {}

    cur = conn.cursor()

    # Get all non-dead relationships
    cur.execute("""
        SELECT r.source_id, r.target_id, r.predicate, r.decay_score, r.observation_count
        FROM relationships r
        WHERE r.decay_score > 0.1
    """)
    edges = cur.fetchall()

    # Build adjacency with edge weights (decay * log(obs+1))
    adj = defaultdict(list)  # entity_id -> [(neighbor_id, weight, predicate)]
    edge_count = defaultdict(int)

    for src, tgt, pred, decay, obs in edges:
        import math
        weight = decay * math.log2(max(obs, 1) + 1)
        adj[src].append((tgt, weight, pred))
        adj[tgt].append((src, weight, pred))
        edge_count[src] += 1
        edge_count[tgt] += 1

    # Filter to entities with enough connections
    qualified = {eid for eid, cnt in edge_count.items() if cnt >= MIN_EDGES_FOR_CLUSTER}
    filtered_adj = {
        eid: [(n, w, p) for n, w, p in neighbors if n in qualified]
        for eid, neighbors in adj.items()
        if eid in qualified
    }

    # Get entity info
    if qualified:
        cur.execute(
            "SELECT id, name, type FROM entities WHERE id = ANY(%s)",
            (list(qualified),)
        )
        entity_info = {r[0]: {"name": r[1], "type": r[2]} for r in cur.fetchall()}
    else:
        entity_info = {}

    return filtered_adj, entity_info


def detect_communities() -> dict[int, list[int]]:
    """
    Label propagation community detection.

    Each node starts with its own label. Iteratively, each node adopts
    the most common label among its neighbors (weighted by edge weight).
    Converges when no labels change.

    Returns: {cluster_id: [entity_id, ...]}
    """
    adj, entity_info = _build_adjacency()
    if not adj:
        _log("No qualified entities for clustering")
        return {}

    import random

    # Initialize: each entity is its own cluster
    labels = {eid: eid for eid in adj}

    for iteration in range(CLUSTER_MAX_ITERATIONS):
        changed = 0
        # Process nodes in random order to avoid bias
        nodes = list(adj.keys())
        random.shuffle(nodes)

        for node in nodes:
            neighbors = adj.get(node, [])
            if not neighbors:
                continue

            # Count weighted votes for each label
            label_weights = defaultdict(float)
            for neighbor, weight, _ in neighbors:
                if neighbor in labels:
                    label_weights[labels[neighbor]] += weight

            if not label_weights:
                continue

            # Pick label with highest weight (break ties randomly)
            max_weight = max(label_weights.values())
            best_labels = [l for l, w in label_weights.items() if w == max_weight]
            best = random.choice(best_labels)

            if labels[node] != best:
                labels[node] = best
                changed += 1

        if changed == 0:
            _log(f"Label propagation converged at iteration {iteration + 1}")
            break
    else:
        _log(f"Label propagation hit max iterations ({CLUSTER_MAX_ITERATIONS})")

    # Group by cluster label
    clusters = defaultdict(list)
    for eid, label in labels.items():
        clusters[label].append(eid)

    # Filter out tiny clusters
    clusters = {
        cid: members for cid, members in clusters.items()
        if len(members) >= MIN_CLUSTER_SIZE
    }

    _log(f"Detected {len(clusters)} clusters from {len(adj)} entities")
    return clusters


def _generate_cluster_label(entity_names: list[str], predicates: list[str]) -> str:
    """Generate a human-readable cluster label from its top entities and predicates."""
    # Simple heuristic: use top 2-3 entity names
    top = entity_names[:3]
    return " / ".join(top)


def store_clusters(clusters: dict[int, list[int]]):
    """
    Persist detected clusters to entity_clusters + cluster_meta tables.
    Full replace — drops old assignments and rewrites.
    """
    conn = db.get_pg()
    if not conn:
        return

    cur = conn.cursor()

    # Clear old clusters
    cur.execute("DELETE FROM entity_clusters")
    cur.execute("DELETE FROM cluster_meta")

    cluster_num = 0
    for _label_id, members in clusters.items():
        cluster_num += 1

        # Get entity names for this cluster
        cur.execute(
            "SELECT id, name, type FROM entities WHERE id = ANY(%s) ORDER BY name",
            (members,)
        )
        entities = cur.fetchall()
        entity_names = [e[1] for e in entities]

        # Get top predicates within this cluster
        cur.execute("""
            SELECT predicate, COUNT(*) as cnt
            FROM relationships
            WHERE source_id = ANY(%s) AND target_id = ANY(%s)
            GROUP BY predicate ORDER BY cnt DESC LIMIT 5
        """, (members, members))
        top_preds = [r[0] for r in cur.fetchall()]

        # Generate label
        label = _generate_cluster_label(entity_names, top_preds)

        # Insert cluster assignments
        for eid in members:
            cur.execute(
                """INSERT INTO entity_clusters (cluster_id, entity_id, cluster_label, confidence)
                   VALUES (%s, %s, %s, 1.0)
                   ON CONFLICT (entity_id) DO UPDATE SET
                     cluster_id = EXCLUDED.cluster_id,
                     cluster_label = EXCLUDED.cluster_label,
                     updated_at = NOW()
                """,
                (cluster_num, eid, label)
            )

        # Insert cluster metadata
        cur.execute(
            """INSERT INTO cluster_meta
               (cluster_id, label, entity_count, top_entities, top_predicates)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (cluster_id) DO UPDATE SET
                 label = EXCLUDED.label,
                 entity_count = EXCLUDED.entity_count,
                 top_entities = EXCLUDED.top_entities,
                 top_predicates = EXCLUDED.top_predicates,
                 updated_at = NOW()
            """,
            (cluster_num, label, len(members),
             entity_names[:5], top_preds[:5])
        )

    conn.commit()
    _log(f"Stored {cluster_num} clusters ({sum(len(m) for m in clusters.values())} entities)")


def run_clustering():
    """Full clustering pipeline: detect → label → store."""
    clusters = detect_communities()
    if clusters:
        store_clusters(clusters)
    return clusters


# ═══════════════════════════════════════════════════════════════════════════════
# Cluster-aware retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def get_cluster_for_entity(entity_name: str) -> dict | None:
    """
    Given an entity name, return its cluster info including all members.

    Returns:
        {
            "cluster_id": int,
            "label": str,
            "members": [{"name": str, "type": str, "id": int}, ...],
            "internal_edges": [{"source": str, "predicate": str, "target": str, "decay": float}, ...],
        }
    """
    conn = db.get_pg()
    if not conn:
        return None

    cur = conn.cursor()

    # Find the entity
    cur.execute(
        "SELECT id FROM entities WHERE name ILIKE %s LIMIT 1",
        (entity_name,)
    )
    row = cur.fetchone()
    if not row:
        return None
    entity_id = row[0]

    # Find its cluster
    cur.execute(
        "SELECT cluster_id, cluster_label FROM entity_clusters WHERE entity_id = %s",
        (entity_id,)
    )
    row = cur.fetchone()
    if not row:
        return None
    cluster_id, cluster_label = row

    # Get all members
    cur.execute("""
        SELECT e.id, e.name, e.type
        FROM entity_clusters ec
        JOIN entities e ON e.id = ec.entity_id
        WHERE ec.cluster_id = %s
        ORDER BY e.name
    """, (cluster_id,))
    members = [{"id": r[0], "name": r[1], "type": r[2]} for r in cur.fetchall()]
    member_ids = [m["id"] for m in members]

    # Get internal edges (within cluster)
    cur.execute("""
        SELECT e1.name, r.predicate, e2.name, r.decay_score, r.observation_count
        FROM relationships r
        JOIN entities e1 ON e1.id = r.source_id
        JOIN entities e2 ON e2.id = r.target_id
        WHERE r.source_id = ANY(%s) AND r.target_id = ANY(%s)
          AND r.decay_score > 0.1
        ORDER BY r.decay_score DESC, r.observation_count DESC
        LIMIT 30
    """, (member_ids, member_ids))
    edges = [
        {"source": r[0], "predicate": r[1], "target": r[2],
         "decay": round(r[3], 3), "observations": r[4]}
        for r in cur.fetchall()
    ]

    return {
        "cluster_id": cluster_id,
        "label": cluster_label,
        "members": members,
        "internal_edges": edges,
    }


def get_cluster_context(entity_names: list[str]) -> str:
    """
    Build a <CLUSTER_CONTEXT> block for prompt injection.
    Given entity names, finds their clusters and formats as readable context.
    """
    if not entity_names:
        return ""

    seen_clusters = set()
    blocks = []

    for name in entity_names:
        cluster = get_cluster_for_entity(name)
        if not cluster or cluster["cluster_id"] in seen_clusters:
            continue
        seen_clusters.add(cluster["cluster_id"])

        member_list = ", ".join(
            f"{m['name']} ({m['type']})" for m in cluster["members"][:8]
        )
        extra = len(cluster["members"]) - 8
        if extra > 0:
            member_list += f", +{extra} more"

        edge_lines = []
        for e in cluster["internal_edges"][:10]:
            decay_tag = "" if e["decay"] > 0.7 else f" [decay={e['decay']}]"
            edge_lines.append(
                f"  {e['source']} —{e['predicate']}→ {e['target']}"
                f" (×{e['observations']}){decay_tag}"
            )

        block = f"Cluster: {cluster['label']}\n"
        block += f"Members: {member_list}\n"
        if edge_lines:
            block += "Relationships:\n" + "\n".join(edge_lines)

        blocks.append(block)

    if not blocks:
        return ""

    return "<CLUSTER_CONTEXT>\n" + "\n\n".join(blocks) + "\n</CLUSTER_CONTEXT>"


# ═══════════════════════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════════════════════

def show_stats():
    """Print current graph evolution statistics."""
    conn = db.get_pg()
    if not conn:
        print("ERROR: No PG connection")
        return

    cur = conn.cursor()

    # Relationship stats
    cur.execute("SELECT COUNT(*) FROM relationships")
    total_rels = cur.fetchone()[0]

    cur.execute("""
        SELECT
            AVG(observation_count) as avg_obs,
            MAX(observation_count) as max_obs,
            AVG(decay_score) as avg_decay,
            MIN(decay_score) as min_decay
        FROM relationships
        WHERE observation_count IS NOT NULL
    """)
    row = cur.fetchone()
    avg_obs = row[0] or 0
    max_obs = row[1] or 0
    avg_decay = row[2] or 0
    min_decay = row[3] or 0

    print(f"\n  Graph Evolution Stats")
    print(f"  {'='*50}")
    print(f"  Total relationships:  {total_rels}")
    print(f"  Avg observations:     {avg_obs:.1f}")
    print(f"  Max observations:     {max_obs}")
    print(f"  Avg decay score:      {avg_decay:.3f}")
    print(f"  Min decay score:      {min_decay:.3f}")
    print(f"  Decay half-life:      {DECAY_HALF_LIFE_DAYS} days")

    # Decay distribution
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE decay_score > 0.8) as strong,
            COUNT(*) FILTER (WHERE decay_score BETWEEN 0.4 AND 0.8) as moderate,
            COUNT(*) FILTER (WHERE decay_score BETWEEN 0.1 AND 0.4) as weak,
            COUNT(*) FILTER (WHERE decay_score < 0.1) as dead
        FROM relationships
    """)
    strong, moderate, weak, dead = cur.fetchone()
    print(f"\n  Decay distribution:")
    print(f"    Strong  (>0.8):   {strong}")
    print(f"    Moderate (0.4-0.8): {moderate}")
    print(f"    Weak    (0.1-0.4): {weak}")
    print(f"    Dead    (<0.1):   {dead}")

    # Most reinforced relationships
    cur.execute("""
        SELECT e1.name, r.predicate, e2.name, r.observation_count, r.decay_score
        FROM relationships r
        JOIN entities e1 ON e1.id = r.source_id
        JOIN entities e2 ON e2.id = r.target_id
        WHERE r.observation_count IS NOT NULL
        ORDER BY r.observation_count DESC
        LIMIT 10
    """)
    rows = cur.fetchall()
    if rows:
        print(f"\n  Most reinforced edges:")
        for src, pred, tgt, obs, decay in rows:
            print(f"    {src} —{pred}→ {tgt}  (×{obs}, decay={decay:.2f})")

    # Cluster stats
    cur.execute("SELECT COUNT(DISTINCT cluster_id) FROM entity_clusters")
    num_clusters = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM entity_clusters")
    clustered_entities = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM entities")
    total_entities = cur.fetchone()[0]

    print(f"\n  Clusters:")
    print(f"    Total clusters:     {num_clusters}")
    print(f"    Clustered entities: {clustered_entities}/{total_entities} "
          f"({100*clustered_entities/max(total_entities,1):.0f}%)")

    if num_clusters > 0:
        cur.execute("""
            SELECT cluster_id, label, entity_count, top_entities[:3]
            FROM cluster_meta
            ORDER BY entity_count DESC
            LIMIT 10
        """)
        rows = cur.fetchall()
        if rows:
            print(f"\n  Top clusters:")
            for cid, label, count, top in rows:
                top_str = ", ".join(top) if top else ""
                print(f"    #{cid} [{count} entities] {label}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Daemon handlers
# ═══════════════════════════════════════════════════════════════════════════════

def handle_cluster_context(params: dict) -> dict:
    """Daemon handler: get cluster context for entity names."""
    entities = params.get("entities", [])
    if isinstance(entities, str):
        entities = [entities]
    if not entities:
        return {"ok": False, "error": "Missing 'entities' parameter"}

    ctx = get_cluster_context(entities)
    return {"ok": True, "context": ctx}


def handle_cluster_for_entity(params: dict) -> dict:
    """Daemon handler: get full cluster info for one entity."""
    name = params.get("entity", "")
    if not name:
        return {"ok": False, "error": "Missing 'entity' parameter"}

    cluster = get_cluster_for_entity(name)
    if not cluster:
        return {"ok": True, "cluster": None, "message": f"No cluster found for '{name}'"}

    return {"ok": True, "cluster": cluster}


def handle_graph_stats(params: dict) -> dict:
    """Daemon handler: return graph evolution stats as JSON."""
    conn = db.get_pg()
    if not conn:
        return {"ok": False, "error": "No PG connection"}

    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM relationships")
    total_rels = cur.fetchone()[0]

    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE decay_score > 0.8) as strong,
            COUNT(*) FILTER (WHERE decay_score BETWEEN 0.4 AND 0.8) as moderate,
            COUNT(*) FILTER (WHERE decay_score BETWEEN 0.1 AND 0.4) as weak,
            COUNT(*) FILTER (WHERE decay_score < 0.1) as dead
        FROM relationships
    """)
    strong, moderate, weak, dead = cur.fetchone()

    cur.execute("SELECT COUNT(DISTINCT cluster_id) FROM entity_clusters")
    num_clusters = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM entity_clusters")
    clustered = cur.fetchone()[0]

    return {
        "ok": True,
        "relationships": total_rels,
        "decay": {"strong": strong, "moderate": moderate, "weak": weak, "dead": dead},
        "clusters": num_clusters,
        "clustered_entities": clustered,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Graph Evolution — Phase 7A")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("migrate", help="Apply schema migrations")
    sub.add_parser("decay", help="Recompute decay scores")
    sub.add_parser("cluster", help="Run community detection + store")
    sub.add_parser("stats", help="Show graph evolution stats")

    p_ctx = sub.add_parser("context", help="Get cluster context for entities")
    p_ctx.add_argument("entities", nargs="+", help="Entity names")

    p_lookup = sub.add_parser("lookup", help="Look up cluster for one entity")
    p_lookup.add_argument("entity", help="Entity name")

    sub.add_parser("full", help="Run full pipeline: decay → cluster → stats")

    args = parser.parse_args()

    if args.command == "migrate":
        migrate()

    elif args.command == "decay":
        migrate()  # ensure columns exist
        decay_relationships()

    elif args.command == "cluster":
        migrate()  # ensure tables exist
        run_clustering()

    elif args.command == "stats":
        show_stats()

    elif args.command == "context":
        ctx = get_cluster_context(args.entities)
        print(ctx if ctx else "No cluster context found")

    elif args.command == "lookup":
        cluster = get_cluster_for_entity(args.entity)
        if cluster:
            print(json.dumps(cluster, indent=2, default=str))
        else:
            print(f"No cluster found for '{args.entity}'")

    elif args.command == "full":
        _log("Full pipeline: migrate → decay → cluster → stats")
        migrate()
        decay_relationships()
        run_clustering()
        show_stats()

    else:
        parser.print_help()
