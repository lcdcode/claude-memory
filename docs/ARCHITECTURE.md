# claude-memory: Architecture and Data Model

A technical reference for the local-only persistent memory MCP server. This document
describes how text becomes a searchable memory, how retrieval ranks results, and how the
data is physically laid out in PostgreSQL.

Everything here is derived from the source in `src/` and the live schema in
`sql/schema.sql`. Nothing in this system contacts the network beyond `127.0.0.1`.

---

## 1. General Overview

claude-memory is a [Model Context Protocol](https://modelcontextprotocol.io) server that
gives an LLM client a durable, searchable memory store. It runs as a stdio subprocess that
the client launches on demand (`src/server.py` -> `mcp.run(transport="stdio")`).

Three local components cooperate:

| Component | Role | Where |
| --- | --- | --- |
| **Ollama** | Turns text into a 768-dim embedding vector (`nomic-embed-text`) | `127.0.0.1:11434` |
| **PostgreSQL + pgvector + pg_trgm** | Stores memories and runs vector/trigram search | `127.0.0.1:5432` |
| **Python MCP server** | Orchestrates embedding, search, fusion, and re-ranking | `src/*.py` |

The defining design choice is the **retrieval pipeline**. A naive memory store would do one
vector similarity query and return the top-k. This system instead layers four ideas:

1. **Semantic search** via pgvector cosine similarity (meaning, not keywords).
2. **Lexical search** via pg_trgm trigram similarity (exact tokens: error codes, names).
3. **Reciprocal Rank Fusion (RRF)** to merge those two ranked lists into one.
4. **ACT-R cognitive re-ranking**, a model from cognitive psychology that scores each
   memory by how "active" it would be in human declarative memory: a blend of recency,
   access frequency, semantic relevance, and associative priming.

A companion **strategic forgetting** process periodically demotes stale memories to
`dormant` or `forgotten` status without deleting them.

### End-to-end data flow

```
store_memory(content, ...)
   content --> Ollama --> [768 floats] --> INSERT into memories (embedding column)

retrieve_memories(query, ...)
   query --> Ollama --> [768 floats]
                            |
              +-------------+-------------+
              v                           v
        VECTOR query                 TRIGRAM query
     (cosine, pgvector)            (pg_trgm similarity)
              |                           |
              +------------+--------------+
                           v
              Reciprocal Rank Fusion (k=60)
                           v
              ACT-R re-rank: A(m) = B(m) + w*cos + S(m) + epsilon
                           v
              top-N results + bump access_count / access_timestamps
```

---

## 2. The Embedding Layer

### What an embedding is

`get_embedding()` in `src/server.py` POSTs the text to Ollama's `/api/embeddings`
endpoint and receives a list of 768 floats. That list **is** the memory's semantic
fingerprint. The model (`nomic-embed-text`) is trained so that texts with similar meaning
map to nearby points in 768-dimensional space.

Observed properties of real vectors in this database:

- **Dimensionality**: exactly 768 (must match `VECTOR(768)` in the schema).
- **Dense**: every component is nonzero. No sparsity, unlike TF-IDF/bag-of-words.
- **Roughly zero-centered**: component mean approximately 0, values typically in `[-3, +2]`.
- **Non-unit length**: L2 norm approximately 17 and varies per row. The model does not
  normalize its output, which is precisely why the system compares vectors by **cosine**
  (direction) rather than Euclidean distance (which would be polluted by magnitude).

### SSRF guard

`OLLAMA_HOST` is parsed at startup and the server refuses to run if the hostname is
anything other than `localhost` or `127.0.0.1` (`src/server.py`). This prevents the
embedding call from being redirected to an attacker-controlled host.

### Serialization

pgvector accepts a vector as a bracketed string literal. `format_embedding()` builds
`"[0.21822,-0.05118,...]"` and the SQL casts it with `$1::vector`. On disk pgvector stores
it compactly (4 bytes per dimension, approximately 3 KB per row), independent of the text
rendering you see in `psql`.

---

## 3. The Retrieval Pipeline

### 3.1 Two parallel queries (`src/hybrid_search.py`)

`build_search_queries()` constructs two SQL statements sharing the same WHERE-clause
filters (category, tags, project, user scoping, and forgotten-exclusion).

**Vector query** ranks by cosine similarity. pgvector's `<=>` operator is cosine
*distance*, so similarity is `1 - distance`:

```sql
SELECT id, content, summary, ...,
       1 - (embedding <=> $1::vector) AS sim
FROM memories
WHERE 1 - (embedding <=> $1::vector) >= $2   -- min_similarity, default 0.3
  {filters}
ORDER BY (1 - (embedding <=> $1::vector)) DESC
LIMIT $3;                                      -- prefetch (default 50)
```

**Trigram query** ranks by `pg_trgm` lexical overlap on the concatenated
`content || ' ' || summary`, with a low floor (`>= 0.05`) so it only contributes signal
when there is genuine token overlap:

```sql
SELECT id, content, summary, ...,
       similarity(content || ' ' || COALESCE(summary,''), $1) AS trgm_sim,
       1 - (embedding <=> $2::vector) AS sim
FROM memories
WHERE similarity(content || ' ' || COALESCE(summary,''), $1) >= 0.05
  {filters}
ORDER BY similarity(...) DESC
LIMIT $3;
```

The trigram query is wrapped in a `try/except` in `server.py`: if `pg_trgm` is missing or
errors, the system **degrades gracefully to vector-only** rather than failing the request.

### 3.2 Reciprocal Rank Fusion (`reciprocal_rank_fusion`)

RRF merges the two ranked lists using only each document's **rank position**, not its raw
score, which sidesteps the problem that cosine similarity and trigram similarity are on
incomparable scales. For a document appearing at rank `r` (0-indexed) in a list:

```
contribution = 1 / (k + r)        with k = 60
```

A document's fused score is the sum of its contributions across both lists. Appearing in
both lists is rewarded; the constant `k=60` damps the influence of any single high rank so
no one list dominates. Ties favor the vector list (it is inserted into `doc_store` first).

### 3.3 ACT-R cognitive re-ranking (`src/actr_scoring.py`)

The fused list is re-scored by the ACT-R declarative-memory activation equation:

```
A(m) = B(m) + w * cos_sim(query, m) + S(m) + epsilon
```

**B(m) - base-level activation** (`compute_base_level`). Models the power-law decay of
human memory: each past access contributes, but older accesses contribute less.

```
B(m) = ln( sum_i (t_now - t_i)^(-d) )
```

where `t_i` are the entries in `access_timestamps` (falling back to `created_at` if never
accessed) and `d` is the decay rate (default `0.5`). More-recent and more-frequent access
both raise `B(m)`.

**w * cos_sim - semantic relevance**. The cosine similarity from the vector query, scaled
by `w` (default `11.0`). The weight is large because `cos_sim` lives in `[0,1]` while
`B(m)` can be on the order of single digits; `w` puts them on a comparable footing.

`w` is **adaptive** (`classify_query_type` / `get_adaptive_w`). If the query text, tags, or
category look like debugging (keywords: `error`, `bug`, `traceback`, ... or category
`bugfix`/`error_solution`), `w` is multiplied by `1.5` to favor semantic precision.
Architecture-flavored queries and general queries use `1.0`.

**S(m) - spreading activation** (`compute_spreading_activation`). Optional associative
priming from tags shared between the query and the memory. Rarer shared tags carry more
signal, mirroring the ACT-R "fan effect":

```
S(m) = sum over shared tags of  W * (S - ln(fan_of_tag))
W = 1 / |query_tags|
```

`fan_of_tag` is how many memories carry that tag (computed live in `server.py` via an
`unnest(tags)` aggregate). A tag attached to many memories is less discriminating, so its
contribution shrinks.

**epsilon - noise** (`compute_noise`). Optional Gaussian noise (`sigma` default `1.2`) that
makes retrieval probabilistic rather than perfectly deterministic, matching ACT-R's account
of human recall variability.

**Retrieval threshold**. Memories whose total `A(m)` falls below `tau` (default `-2.0`) are
dropped during scoring. Survivors are sorted by `A(m)` descending and truncated to
`max_results`.

To disable all of this and fall back to classic ranking, set `USE_ACTR_SCORING=false`; the
server then sorts by `sim * importance_score`.

### 3.4 Access bookkeeping

After the final set is chosen, `retrieve_memories` issues one UPDATE that, for each
returned memory: sets `last_accessed_at = NOW()`, increments `access_count`, and appends
`NOW()` to `access_timestamps`. The timestamp array is **capped at 1000 entries** (oldest
dropped via `access_timestamps[2:]`) to bound row growth. These writes are what feed
`B(m)` on future retrievals: retrieving a memory makes it easier to retrieve again.

---

### 3.5 Worked numerical example

To make the scoring concrete, here is the full arithmetic for one real memory in the
database. All inputs are actual stored values; `now` is fixed at `2026-06-04 12:00:00-05:00`
so the result is reproducible.

**Memory** (id `f5e152fb...`, category `feature`):
*"claude-memory MCP server - local Postgres+pgvector+Ollama memory with ACT-R scoring..."*

Stored `access_timestamps` (it has been retrieved three times):

```
2026-06-01 17:08:34.157053-05
2026-06-01 17:12:15.262317-05
2026-06-01 17:12:16.177087-05
```

**Step 1 - base-level activation `B(m) = ln( sum_i ((t_now - t_i) / unit)^(-d) )`, with
`d = 0.5` and `unit = 86400` (one day; see section 3.6).**

Each access is roughly 2.78 days old. We express the age in days, raise to the `-0.5` power
(i.e. `1 / sqrt(days)`), and sum:

| access time | `delta_days` | `(delta_days)^-0.5` |
| --- | ---: | ---: |
| 17:08:34 | 2.7857 | 0.59914 |
| 17:12:15 | 2.7832 | 0.59942 |
| 17:12:16 | 2.7831 | 0.59942 |
| | **sum** | **1.79799** |

```
B(m) = ln(1.79798) = +0.587
```

The base level is mildly positive: three accesses, none older than three days, keep the
memory `active`. (Under the original raw-seconds code this same row scored `B = -5.10` and
would have been wrongly marked `forgotten` - that calibration bug is covered in section 3.6.)

**Step 2 - full activation `A(m) = B(m) + w * cos_sim + S(m) + epsilon`.**

Suppose a query about "local memory server" embeds and yields `cos_sim = 0.79` against this
memory (a real value we observed for a related query). It is a general query, so the
adaptive weight stays at `w = 11.0`. Ignore spreading and noise for now (`S = 0`,
`epsilon = 0`):

```
A(m) = 0.587 + 11.0 * 0.79
     = 0.587 + 8.69
     = 9.28
```

The semantic term dominates (`+8.69` vs a base of `+0.59`): for an active, relevant memory,
*what it is about* matters far more than exactly when it was last touched. Base level breaks
ties and governs long-term retention; semantic similarity drives in-the-moment ranking.

**Step 3 - variations.**

- *Weaker match* (`cos_sim = 0.55`): `A = 0.587 + 11.0*0.55 = 6.64`. Still strongly
  retrievable; the semantic term alone keeps it well above threshold.
- *Debugging query* (adaptive `w = 11.0 * 1.5 = 16.5`, `cos_sim = 0.79`):
  `A = 0.587 + 16.5*0.79 = 13.62`. The 1.5x multiplier more than doubles the effective
  influence of semantic similarity, pushing strong matches far up the ranking for
  error/bug-flavored queries.

**Semantic rescue of a stale memory.** Now take a different memory that has gone 60 days
untouched: a single access gives `B = -0.5 * ln(60) = -2.05`, below the `-2.0` forgetting
boundary, so a forgetting cycle would mark it `forgotten`. If a later query is genuinely
about it (`cos_sim = 0.60`), its activation is `A = -2.05 + 11.0*0.60 = 4.55`, comfortably
above the retrieval threshold `tau = -2.0`. The semantic term *rescues* it: it is surfaced
despite being forgotten, and the resurrection safeguard (section 4) then promotes it back to
`active`. Recency governs default retention; relevance can always override it.

**Step 4 - spreading activation `S(m)`** (if the query carries tags). Suppose the query
supplies two tags and shares one with the memory. Per shared tag the contribution is
`W * (S - ln(fan))` with `W = 1/|query_tags| = 0.5` and `S = 2.0`, where `fan` is how many
memories carry that tag (real counts from this DB):

| shared tag | `fan` | `0.5 * (2.0 - ln(fan))` |
| --- | ---: | ---: |
| `postgres` | 4 | `0.5 * (2.0 - 1.386) = 0.307` |
| `act-r` | 1 | `0.5 * (2.0 - 0.000) = 1.000` |

A rare tag (`act-r`, on one memory) adds a full `+1.0`; a common tag (`postgres`, on four)
adds only `+0.31`. This is the ACT-R **fan effect**: an association that points at many
things is a weak cue, so it is down-weighted by `ln(fan)`. Added to the `cos_sim = 0.79`
general-query case from Step 2 (`A = 9.28`), a shared `act-r` tag would raise `A(m)` to
about `10.28`.

### 3.6 Base-level time unit (calibration)

`B(m)` measures each access age in units of `time_unit_seconds` (config
`ACTR_TIME_UNIT_SECONDS`, default `86400` = one day) before applying the power-law decay.
This is not cosmetic: the status thresholds in `forgetting.py` (`0` and `-2`) and the
retrieval `tau` (`-2`) are only meaningful relative to that unit.

Concretely, for a single-access memory `B = -d * ln(age / unit)`. With `d = 0.5`:

| Time unit | `active` boundary (B=0) | `forgotten` boundary (B=-2) |
| --- | --- | --- |
| seconds (unit = 1) | age = 1 s | age = ~55 s |
| **days (unit = 86400, default)** | age = 1 day | age = ~55 days |

Measuring in raw seconds would push a single-access memory past the forgetting threshold in
under a minute, so a forgetting cycle would mark essentially the entire store `forgotten`.
Changing the unit is a pure constant shift of every `B` by `d * ln(unit)` (about `+5.68`
for one day), so it re-centers the thresholds without distorting the relative ordering of
memories. Tune `ACTR_TIME_UNIT_SECONDS` to match how often memories are realistically
re-accessed in your workflow.

---

## 4. Strategic Forgetting (`src/forgetting.py`)

`memory_forgetting_cycle` recomputes activation for every memory and transitions its
`memory_status`. Important nuance: the cycle classifies on **base-level activation `B(m)`
alone**, not the full `A(m)` (no semantic, spreading, or noise terms - there is no query
during a forgetting pass).

```
B(m) > 0        -> active     (readily retrievable)
-2 < B(m) <= 0  -> dormant    (retained, deprioritized)
B(m) <= -2      -> forgotten  (excluded from default queries, NOT deleted)
```

The computed value is persisted to `actr_activation` and `activation_updated_at`. So the
`actr_activation` column holds `B(m)` as of the last cycle, not the full activation seen
during retrieval.

**Forgotten is not deleted.** Rows remain in the table. Only `delete_memory` removes data.

**Resurrection (avoiding the one-way trap).** Forgetting must be reversible, or a memory
that becomes relevant again could never come back. Two mechanisms ensure recovery:

1. *Semantic rescue.* The vector branch of `retrieve_memories` keeps forgotten rows as
   eligible candidates (only the lexical/trigram branch excludes them by default). A
   forgotten memory with high enough `cos_sim` to clear `min_similarity` and `tau` is
   surfaced, exactly as in the section 3.5 example.
2. *Retrieval as rehearsal.* Any memory returned by `retrieve_memories` is promoted back to
   `active` in the same statement that bumps its access count and timestamps. So the act of
   being retrieved restores retrievability; a memory can only stay forgotten while nothing
   relevant ever asks for it. The next forgetting cycle then recomputes its status from the
   refreshed `access_timestamps`.

Without these, the access-stat update only touches returned rows, and forgotten rows were
never returned - so `B(m)` could only ever decay further. That one-way decay is the trap
these mechanisms close.

**Concurrency.** The cycle runs inside a transaction guarded by
`pg_advisory_xact_lock(42)` and selects rows `FOR UPDATE`, so two concurrent cycles cannot
interleave their status writes.

---

## 5. Database Structure

### 5.1 The `memories` table

Single-table design (`sql/schema.sql`). Columns grouped by purpose:

**Identity and content**

| Column | Type | Notes |
| --- | --- | --- |
| `id` | `UUID` PK | `DEFAULT gen_random_uuid()` |
| `content` | `TEXT NOT NULL` | Full memory text (server caps input at 50000 chars) |
| `summary` | `TEXT` | Short summary; auto-derived from first 150 chars if omitted |
| `category` | `VARCHAR(50) NOT NULL` | CHECK-constrained (see below) |
| `tags` | `TEXT[]` | Default `'{}'`; used for filtering and spreading activation |
| `project_context` | `VARCHAR(500)` | Optional project scoping |
| `embedding` | `VECTOR(768)` | pgvector type; the semantic fingerprint |

**Temporal and usage**

| Column | Type | Notes |
| --- | --- | --- |
| `created_at` | `TIMESTAMPTZ` | `DEFAULT NOW()` |
| `updated_at` | `TIMESTAMPTZ` | `DEFAULT NOW()` |
| `last_accessed_at` | `TIMESTAMPTZ` | Bumped on every retrieval |
| `access_count` | `INTEGER` | Incremented on every retrieval |
| `importance_score` | `DOUBLE PRECISION` | `DEFAULT 0.5`, CHECK in `[0,1]` |
| `access_timestamps` | `TIMESTAMPTZ[]` | Per-access history feeding `B(m)`; capped at 1000 |

**ACT-R and ownership**

| Column | Type | Notes |
| --- | --- | --- |
| `memory_status` | `VARCHAR(10)` | `DEFAULT 'active'`, CHECK in {active, dormant, forgotten} |
| `actr_activation` | `DOUBLE PRECISION` | Last computed `B(m)`; NULL until first forgetting cycle |
| `activation_updated_at` | `TIMESTAMPTZ` | When `actr_activation` was last written |
| `user_id` | `VARCHAR(255)` | `DEFAULT 'default'`; multi-tenant scoping |

### 5.2 Constraints

CHECK constraints (not ENUMs) keep the allowed value sets in sync with `server.py` without
DDL migrations:

- `valid_category`: one of `bugfix, decision, feature, discovery, refactor, change,
  learning, pattern, error_solution, preference`.
- `valid_importance`: `0 <= importance_score <= 1`.
- `valid_memory_status`: one of `active, dormant, forgotten`.

### 5.3 Indexes

| Index | Type | Purpose |
| --- | --- | --- |
| `idx_memories_embedding` | **HNSW** `vector_cosine_ops` | Approximate nearest-neighbor for the vector query |
| `idx_memories_content_trgm` | **GIN** `gin_trgm_ops` | Trigram search on `content` |
| `idx_memories_summary_trgm` | **GIN** `gin_trgm_ops` | Trigram search on `summary` |
| `idx_memories_tags` | **GIN** | `tags @> ...` containment filters |
| `idx_memories_category` | B-tree | Category filter |
| `idx_memories_project` | B-tree | Project filter |
| `idx_memories_status` | B-tree | Status filter / forgotten exclusion |
| `idx_memories_created` | B-tree DESC | `list_memories` ordering |
| `idx_memories_importance` | B-tree DESC | Classic-mode ranking |

**Why HNSW.** Hierarchical Navigable Small World builds a layered proximity graph over the
vectors. An exact nearest-neighbor scan is O(n) per query; HNSW navigates the graph in
roughly O(log n), trading a small, tunable amount of recall for large speedups. It is what
lets the vector query scale past a handful of rows. At the current table size the index is
not yet load-bearing, but the design anticipates growth.

### 5.4 Extensions required

- `vector` (pgvector >= 0.5.0) - the `VECTOR` type, `<=>` operator, HNSW index.
  Untrusted extension: creating it needs a superuser once.
- `pg_trgm` - `similarity()` and `gin_trgm_ops`. Trusted extension.

---

## 6. The MCP Tool Surface

All six tools are defined in `src/server.py` with `@mcp.tool()`. Every tool scopes its
queries by `user_id` (matching `USER_ID` from `.env`, or NULL legacy rows).

| Tool | Purpose | Key behavior |
| --- | --- | --- |
| `store_memory` | Persist a memory | Validates category, clamps importance, embeds content, INSERT |
| `retrieve_memories` | Hybrid + ACT-R search | The full pipeline of section 3; updates access stats |
| `list_memories` | Recent memories, newest first | Pure metadata listing, `ORDER BY created_at DESC` |
| `delete_memory` | Hard-delete one row by UUID | The only path that removes data |
| `memory_stats` | Counts by category/status, most-accessed, avg activation | Read-only aggregates |
| `memory_forgetting_cycle` | Recompute `B(m)`, transition statuses | Advisory-locked transaction |

---

## 7. Configuration Reference (`.env`)

| Variable | Default | Effect |
| --- | --- | --- |
| `PG_HOST` / `PG_PORT` | `127.0.0.1` / `5432` | PostgreSQL connection |
| `PG_DATABASE` | `claude_memory` | Target database |
| `PG_USER` / `PG_PASSWORD` | - | Credentials (empty under trust/peer auth) |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Embedding endpoint (localhost enforced) |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Must produce 768-dim vectors to match schema |
| `USER_ID` | `default` | Owner tag for stored/retrieved memories |
| `USE_ACTR_SCORING` | `true` | Toggle ACT-R re-ranking vs `sim * importance` |
| `ACTR_DECAY_D` | `0.5` | Power-law decay rate `d` in `B(m)` |
| `ACTR_WEIGHT_W` | `11.0` | Base semantic weight `w` (adaptively scaled) |
| `ACTR_NOISE_SIGMA` | `1.2` | Gaussian noise std dev (`0` = deterministic) |
| `ACTR_THRESHOLD_TAU` | `-2.0` | Retrieval cutoff; below this, dropped |
| `ACTR_TIME_UNIT_SECONDS` | `86400` | Recency unit for `B(m)` (1 day); see section 3.6 |
| `ACTR_SPREADING_S` | `2.0` | Max associative strength `S` |
| `ACTR_USE_SPREADING` / `ACTR_USE_NOISE` | `true` | Toggle the `S(m)` / `epsilon` terms |
| `ACTR_PREFETCH_LIMIT` | `50` | Candidate pool size before re-ranking |

All ACT-R floats are range-clamped in `ACTRConfig.from_env()`, so out-of-range or malformed
values fall back safely rather than corrupting the scoring math.

---

## 8. Design Notes and Limitations

- **Data is not portable with the repo.** Schema travels; rows do not. A fresh clone starts
  with an empty table. Cross-machine sync (encrypted `pg_dump`, logical replication, shared
  private DB) is unsolved.
- **No migration tooling.** Schema drift would need Alembic or numbered `sql/migrations/`.
  The CHECK-constraint-over-ENUM choice is a deliberate hedge against needing DDL migrations
  for value-set changes.
- **Embedding dimension is load-bearing.** Switching to a model with a different output
  dimension requires changing `VECTOR(768)` and re-embedding every existing row; old and new
  vectors are not comparable.
- **`actr_activation` lags reality.** It reflects the last forgetting cycle's `B(m)`, not
  the live `A(m)` used during retrieval. Treat it as a status snapshot, not a current score.
