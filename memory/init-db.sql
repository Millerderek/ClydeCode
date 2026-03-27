-- ═══════════════════════════════════════════════════════════════════════════════
-- OpenClaw Memory Metadata — PostgreSQL Schema
-- Stores structured data about memories: keyscores, confidence, entities,
-- contradictions, access patterns, and decay timers
-- ═══════════════════════════════════════════════════════════════════════════════

-- Memory records: one row per memory in Qdrant
CREATE TABLE IF NOT EXISTS memories (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    qdrant_point_id TEXT UNIQUE NOT NULL,           -- Links to Qdrant vector
    collection      TEXT NOT NULL DEFAULT 'default', -- Qdrant collection name
    content_hash    TEXT NOT NULL,                   -- SHA-256 of memory text (dedup)
    summary         TEXT,                            -- Short summary of the memory
    source          TEXT NOT NULL DEFAULT 'auto',    -- 'auto', 'explicit', 'compaction', 'ingestion'
    confidence      TEXT NOT NULL DEFAULT 'inferred', -- 'confirmed', 'direct', 'inferred', 'ambient'
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed   TIMESTAMPTZ,                    -- For recency scoring
    access_count    INTEGER NOT NULL DEFAULT 0,     -- For frequency scoring
    ttl_expires     TIMESTAMPTZ,                    -- Soft delete after this time
    is_deprecated   BOOLEAN NOT NULL DEFAULT FALSE, -- Superseded by newer memory
    deprecated_by   UUID REFERENCES memories(id),   -- What replaced this
    deprecated_reason TEXT,                          -- 'contradiction', 'stale', 'manual'
    impact_category TEXT NOT NULL DEFAULT 'normal'  -- 'critical', 'high', 'normal', 'low'
);

-- Entity tags: entities mentioned in each memory
CREATE TABLE IF NOT EXISTS memory_entities (
    id          SERIAL PRIMARY KEY,
    memory_id   UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    entity_name TEXT NOT NULL,                      -- e.g., 'Dometic', 'Teams', 'SBC'
    entity_type TEXT NOT NULL DEFAULT 'general',    -- 'client', 'technology', 'person', 'project'
    UNIQUE(memory_id, entity_name)
);

-- Keyscores: precomputed relevance scores, refreshed by custodian sub-agent
CREATE TABLE IF NOT EXISTS keyscores (
    id              SERIAL PRIMARY KEY,
    memory_id       UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    recency_score   REAL NOT NULL DEFAULT 1.0,      -- Decays over time
    frequency_score REAL NOT NULL DEFAULT 0.0,      -- Based on access_count
    authority_score REAL NOT NULL DEFAULT 0.5,       -- Based on confidence + corrections
    entity_boost    REAL NOT NULL DEFAULT 0.0,       -- Contextual boost during retrieval
    impact_score    REAL NOT NULL DEFAULT 0.5,       -- From impact_category (0.25-1.0)
    composite_score REAL NOT NULL DEFAULT 0.5,       -- Weighted combination (v2 with impact)
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(memory_id)
);

-- Contradictions: tracked conflicts between memories
CREATE TABLE IF NOT EXISTS contradictions (
    id              SERIAL PRIMARY KEY,
    memory_a_id     UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    memory_b_id     UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    detected_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved        BOOLEAN NOT NULL DEFAULT FALSE,
    resolution      TEXT,                            -- 'a_wins', 'b_wins', 'both_deprecated', 'user_resolved'
    resolved_at     TIMESTAMPTZ,
    UNIQUE(memory_a_id, memory_b_id)
);

-- Feedback: user corrections and relevance signals
CREATE TABLE IF NOT EXISTS memory_feedback (
    id          SERIAL PRIMARY KEY,
    memory_id   UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    feedback    TEXT NOT NULL,                       -- 'correct', 'incorrect', 'irrelevant', 'useful'
    context     TEXT,                                -- What was being discussed
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Sub-agent job log: track custodian/ingestion runs
CREATE TABLE IF NOT EXISTS agent_jobs (
    id          SERIAL PRIMARY KEY,
    agent_name  TEXT NOT NULL,                      -- 'custodian', 'ingestion', 'retrieval'
    job_type    TEXT NOT NULL,                      -- 'compaction', 'decay', 'contradiction_scan', etc.
    started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status      TEXT NOT NULL DEFAULT 'running',    -- 'running', 'completed', 'failed'
    details     JSONB,                              -- Arbitrary job metadata
    memories_affected INTEGER DEFAULT 0
);

-- ═══════════════════════════════════════════════════════════════════════════════
-- Indexes for fast lookups
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE INDEX IF NOT EXISTS idx_memories_collection ON memories(collection);
CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(last_accessed DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_memories_ttl ON memories(ttl_expires) WHERE ttl_expires IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_deprecated ON memories(is_deprecated) WHERE is_deprecated = FALSE;
CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash);

CREATE INDEX IF NOT EXISTS idx_entities_name ON memory_entities(entity_name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON memory_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_memory ON memory_entities(memory_id);

CREATE INDEX IF NOT EXISTS idx_keyscores_composite ON keyscores(composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_keyscores_memory ON keyscores(memory_id);

CREATE INDEX IF NOT EXISTS idx_contradictions_unresolved ON contradictions(resolved) WHERE resolved = FALSE;

CREATE INDEX IF NOT EXISTS idx_feedback_memory ON memory_feedback(memory_id);

CREATE INDEX IF NOT EXISTS idx_jobs_agent ON agent_jobs(agent_name, started_at DESC);

-- ═══════════════════════════════════════════════════════════════════════════════
-- Helper functions
-- ═══════════════════════════════════════════════════════════════════════════════

-- Auto-update updated_at on memory changes
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS memories_updated_at ON memories;
CREATE TRIGGER memories_updated_at
    BEFORE UPDATE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

-- Recency decay function: exponential decay over days
-- score = exp(-decay_rate * days_since_last_access)
CREATE OR REPLACE FUNCTION compute_recency_score(last_access TIMESTAMPTZ, decay_rate REAL DEFAULT 0.05)
RETURNS REAL AS $$
BEGIN
    IF last_access IS NULL THEN
        RETURN 0.1;
    END IF;
    RETURN EXP(-decay_rate * EXTRACT(EPOCH FROM (NOW() - last_access)) / 86400.0);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Frequency score: logarithmic scaling of access count
CREATE OR REPLACE FUNCTION compute_frequency_score(accesses INTEGER)
RETURNS REAL AS $$
BEGIN
    IF accesses <= 0 THEN
        RETURN 0.0;
    END IF;
    RETURN LEAST(1.0, LN(accesses + 1) / LN(20));  -- Saturates around 20 accesses
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Composite keyscore v1 (legacy, kept for compatibility)
CREATE OR REPLACE FUNCTION compute_composite_score(
    recency REAL, frequency REAL, authority REAL, entity_boost REAL
)
RETURNS REAL AS $$
BEGIN
    RETURN (0.35 * recency) + (0.20 * frequency) + (0.30 * authority) + (0.15 * entity_boost);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Composite keyscore v2: includes impact weight as multiplier
-- Base weights: recency=0.30, frequency=0.20, authority=0.30, entity=0.15, impact=0.05
-- Impact multiplier: low=0.8, normal=1.0, high=1.3, critical=1.5
CREATE OR REPLACE FUNCTION compute_composite_score_v2(
    recency REAL, frequency REAL, authority REAL, entity_boost REAL, impact REAL
)
RETURNS REAL AS $$
DECLARE
    base REAL;
    weight REAL;
BEGIN
    base := (0.30 * recency) + (0.20 * frequency) + (0.30 * authority) + (0.15 * entity_boost) + (0.05 * impact);
    weight := 0.4 + (impact * 1.1);
    weight := GREATEST(0.8, LEAST(1.5, weight));
    RETURN weight * base;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Map impact category text to numeric score
CREATE OR REPLACE FUNCTION impact_category_to_score(cat TEXT)
RETURNS REAL AS $$
BEGIN
    RETURN CASE cat
        WHEN 'critical' THEN 1.0
        WHEN 'high'     THEN 0.75
        WHEN 'normal'   THEN 0.5
        WHEN 'low'      THEN 0.25
        ELSE 0.5
    END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ═══════════════════════════════════════════════════════════════════════════════
-- Views for common queries
-- ═══════════════════════════════════════════════════════════════════════════════

-- Active memories with their current keyscores
CREATE OR REPLACE VIEW active_memories AS
SELECT
    m.id,
    m.qdrant_point_id,
    m.collection,
    m.summary,
    m.source,
    m.confidence,
    m.impact_category,
    m.created_at,
    m.last_accessed,
    m.access_count,
    k.recency_score,
    k.frequency_score,
    k.authority_score,
    k.impact_score,
    k.composite_score
FROM memories m
LEFT JOIN keyscores k ON k.memory_id = m.id
WHERE m.is_deprecated = FALSE
  AND (m.ttl_expires IS NULL OR m.ttl_expires > NOW())
ORDER BY k.composite_score DESC NULLS LAST;

-- Memories needing attention (stale, contradicted, low-confidence)
CREATE OR REPLACE VIEW memories_needing_review AS
SELECT m.id, m.summary, m.confidence, m.last_accessed, m.access_count,
       k.composite_score,
       CASE
           WHEN m.last_accessed < NOW() - INTERVAL '30 days' THEN 'stale'
           WHEN m.confidence = 'ambient' AND m.access_count < 2 THEN 'low_confidence'
           WHEN EXISTS (SELECT 1 FROM contradictions c WHERE (c.memory_a_id = m.id OR c.memory_b_id = m.id) AND c.resolved = FALSE) THEN 'contradicted'
       END AS issue
FROM memories m
LEFT JOIN keyscores k ON k.memory_id = m.id
WHERE m.is_deprecated = FALSE
  AND (
      m.last_accessed < NOW() - INTERVAL '30 days'
      OR (m.confidence = 'ambient' AND m.access_count < 2)
      OR EXISTS (SELECT 1 FROM contradictions c WHERE (c.memory_a_id = m.id OR c.memory_b_id = m.id) AND c.resolved = FALSE)
  );
