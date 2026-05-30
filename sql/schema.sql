-- claude-memory schema bootstrap
-- Idempotent: safe to re-run. Apply against the target database (default: claude_memory).
--
-- Assumes:
--   * PostgreSQL >= 13 (for built-in gen_random_uuid)
--   * pgvector >= 0.5.0 (for HNSW + vector_cosine_ops)
--   * pg_trgm available (standard contrib)
--
-- Embedding dimension is 768, matching the default EMBEDDING_MODEL=nomic-embed-text.
-- If you switch embedding models, change VECTOR(768) and re-embed existing rows.
--
-- Allowed category and memory_status values are enforced via CHECK constraints
-- rather than ENUMs to stay in sync with src/server.py without DDL migrations.

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS memories (
   id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
   content                TEXT NOT NULL,
   summary                TEXT,
   category               VARCHAR(50) NOT NULL,
   tags                   TEXT[] DEFAULT '{}'::text[],
   project_context        VARCHAR(500),
   embedding              VECTOR(768),

   created_at             TIMESTAMPTZ DEFAULT NOW(),
   updated_at             TIMESTAMPTZ DEFAULT NOW(),
   last_accessed_at       TIMESTAMPTZ DEFAULT NOW(),
   access_count           INTEGER DEFAULT 0,
   importance_score       DOUBLE PRECISION DEFAULT 0.5,
   access_timestamps      TIMESTAMPTZ[] DEFAULT '{}'::timestamptz[],

   memory_status          VARCHAR(10) DEFAULT 'active',
   actr_activation        DOUBLE PRECISION,
   activation_updated_at  TIMESTAMPTZ,
   user_id                VARCHAR(255) DEFAULT 'default',

   CONSTRAINT valid_category CHECK (category IN (
      'bugfix', 'decision', 'feature', 'discovery', 'refactor',
      'change', 'learning', 'pattern', 'error_solution', 'preference'
   )),
   CONSTRAINT valid_importance CHECK (
      importance_score >= 0::double precision
      AND importance_score <= 1::double precision
   ),
   CONSTRAINT valid_memory_status CHECK (memory_status IN (
      'active', 'dormant', 'forgotten'
   ))
);

-- Vector ANN index (cosine). pgvector defaults for m/ef_construction; tune for scale.
CREATE INDEX IF NOT EXISTS idx_memories_embedding
   ON memories
   USING hnsw (embedding vector_cosine_ops);

-- Per-column trigram indexes. The hybrid_search query concatenates content+summary,
-- but per-column GIN indexes still help the planner on the dominant column.
CREATE INDEX IF NOT EXISTS idx_memories_content_trgm
   ON memories USING gin (content gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_memories_summary_trgm
   ON memories USING gin (summary gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_memories_tags
   ON memories USING gin (tags);

CREATE INDEX IF NOT EXISTS idx_memories_category
   ON memories (category);

CREATE INDEX IF NOT EXISTS idx_memories_project
   ON memories (project_context);

CREATE INDEX IF NOT EXISTS idx_memories_status
   ON memories (memory_status);

CREATE INDEX IF NOT EXISTS idx_memories_created
   ON memories (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memories_importance
   ON memories (importance_score DESC);

COMMIT;
