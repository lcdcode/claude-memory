# claude-memory

A local-only, persistent memory MCP server for Claude Code (and any MCP client).
Memories are embedded with a local Ollama model, stored in PostgreSQL, and retrieved
with hybrid semantic + lexical search re-ranked by an ACT-R cognitive activation model.
Nothing leaves the machine: no cloud APIs, no tokens spent on embeddings.

This is a **lite, headless variant** - the MCP server only, with no web UI.

## Attribution

This project is a lite reimplementation inspired by
[MrZzE00/MCP-Claude-mem-local](https://github.com/MrZzE00/MCP-Claude-mem-local) (MIT licensed).

- The core ideas are shared: local Postgres + pgvector storage, Ollama embeddings,
  ACT-R cognitive scoring, hybrid retrieval, and strategic forgetting.
- This variant is **headless**. Upstream ships a web interface for browsing, searching,
  and filtering memories; this version exposes those capabilities only through MCP tools.
- This is an independent implementation rather than a fork, so the source code, schema,
  and tunable parameters differ. It was assembled from first principles against the same
  cognitive-science and IR foundations (see Design below).

## Features

- **Six MCP tools**: `store_memory`, `retrieve_memories`, `list_memories`,
  `delete_memory`, `memory_stats`, `memory_forgetting_cycle`.
- **Local embeddings** via Ollama (`nomic-embed-text`, 768-dim). The Ollama host is
  pinned to localhost in `server.py` as an SSRF guard.
- **Hybrid retrieval**: pgvector cosine similarity fused with pg_trgm trigram matching
  via Reciprocal Rank Fusion (k=60). Trigram failures degrade gracefully to vector-only.
- **ACT-R cognitive re-ranking**: results are scored by an activation function combining
  base-level decay (recency + frequency of access), semantic similarity, and optional
  associative spreading from query tags, plus noise. Fully tunable via `.env`.
- **Strategic forgetting**: `memory_forgetting_cycle` recomputes activation for every
  memory and sets its status - `active` (A>0), `dormant` (-2<A<=0), or `forgotten` (A<=-2).
  Forgotten rows stay in the database but are excluded from default queries.
- **Categories** (CHECK-constrained): `bugfix`, `decision`, `feature`, `discovery`,
  `refactor`, `change`, `learning`, `pattern`, `error_solution`, `preference`.
- **Filtering** by category, tags, project context, and minimum similarity.
- **Per-user scoping** via `USER_ID`, plus access-count and access-timestamp tracking.

## Design

The activation score is the ACT-R declarative-memory equation:

```
A(m) = B(m) + w * cos_sim(query, m) + S(m) + epsilon
```

- `B(m)` - base-level activation from access recency and frequency (decay `d`).
- `w * cos_sim` - semantic relevance weight.
- `S(m)` - associative spreading from shared tags (optional).
- `epsilon` - logistic noise (optional).

Retrieval prefetches a wider candidate set (`ACTR_PREFETCH_LIMIT`), fuses the vector and
trigram rankings with RRF, then ACT-R re-ranks and truncates to `max_results`. Set
`USE_ACTR_SCORING=false` to fall back to classic `cos_sim * importance` ordering.

## Requirements

- PostgreSQL >= 13 (>= 14 recommended) with the `pgvector` (>= 0.5.0) and `pg_trgm` extensions.
- [Ollama](https://ollama.com) running locally with an embedding model pulled:
  `ollama pull nomic-embed-text`.
- Python 3.10+ (the project uses a `venv`).

## Setup

```sh
# 1. Python dependencies
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# edit .env: set PG_USER and USER_ID at minimum

# 3. Pull the embedding model
ollama pull nomic-embed-text

# 4. Create the database and apply the schema (idempotent)
scripts/init_db.sh
```

### Configuration (`.env`)

| Variable | Default | Purpose |
| --- | --- | --- |
| `PG_HOST` / `PG_PORT` | `127.0.0.1` / `5432` | PostgreSQL connection. |
| `PG_DATABASE` | `claude_memory` | Target database. |
| `PG_USER` / `PG_PASSWORD` | - | DB credentials (peer/trust auth leaves these empty). |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama endpoint (localhost-only, enforced). |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model; 768-dim must match the schema. |
| `USER_ID` | `default` | Owner tag for stored/retrieved memories. |
| `USE_ACTR_SCORING` | `true` | Toggle ACT-R re-ranking. |
| `ACTR_DECAY_D` | `0.5` | Base-level decay rate. |
| `ACTR_WEIGHT_W` | `11.0` | Semantic similarity weight. |
| `ACTR_NOISE_SIGMA` | `1.2` | Logistic noise scale. |
| `ACTR_THRESHOLD_TAU` | `-2.0` | Forgetting threshold. |
| `ACTR_SPREADING_S` | `2.0` | Associative spreading strength. |
| `ACTR_USE_SPREADING` / `ACTR_USE_NOISE` | `true` | Toggle those terms. |
| `ACTR_PREFETCH_LIMIT` | `50` | Candidate pool size before re-ranking. |

> If you change `EMBEDDING_MODEL` to one with a different dimension, update `VECTOR(768)`
> in `sql/schema.sql` and re-embed existing rows.

## Usage

Register the server with your MCP client. For Claude Code (`.mcp.json`):

```json
{
  "mcpServers": {
    "claude-memory": {
      "type": "stdio",
      "command": "/absolute/path/to/claude-memory/venv/bin/python",
      "args": ["/absolute/path/to/claude-memory/src/server.py"],
      "env": {}
    }
  }
}
```

The server speaks MCP over stdio; the client launches it on demand. Make sure PostgreSQL
and Ollama are running first (see `run-claude-example.sh` for an example launcher that starts
both, then execs `claude`).

### Tools

| Tool | Purpose |
| --- | --- |
| `store_memory(content, category, summary?, tags?, importance?, project?)` | Persist a memory; embeds the content. |
| `retrieve_memories(query, max_results?, category?, min_similarity?, tags?, project?, include_forgotten?)` | Hybrid + ACT-R semantic search. |
| `list_memories(limit?, category?, tags?, project?)` | List recent memories, newest first. |
| `delete_memory(memory_id)` | Delete one memory by UUID. |
| `memory_stats()` | Counts by category and status, most-accessed, average activation. |
| `memory_forgetting_cycle()` | Recompute activation and update active/dormant/forgotten status. |

## Project layout

```
src/
  server.py          MCP server, tool definitions, DB pool, Ollama embedding
  actr_scoring.py    ACT-R activation scoring and ranking
  hybrid_search.py   vector + trigram query builders, RRF fusion
  forgetting.py      strategic forgetting cycle (status transitions)
sql/schema.sql       idempotent DDL (table, extensions, indexes, CHECK constraints)
scripts/init_db.sh   create database + apply schema
.env.example         configuration template
```

## Notes and limitations

- **Memory rows do not travel with the repo.** A fresh clone starts with an empty
  `memories` table; the schema is portable, the data is not. Cross-machine sync
  (encrypted `pg_dump`, logical replication, or a shared private DB) is not yet solved.
- **No migration tooling.** If the schema drifts, add Alembic or numbered
  `sql/migrations/` files.
- Categories and `memory_status` are enforced with CHECK constraints (not ENUMs) so the
  schema stays in sync with `server.py` without DDL migrations.

## License

This lite variant follows the upstream MIT license. See the
[original project](https://github.com/MrZzE00/MCP-Claude-mem-local) for the canonical terms.
