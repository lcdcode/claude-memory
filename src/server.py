#!/usr/bin/env python3
"""claude-memory — MCP server for persistent local memory with semantic search"""

import logging
import os
import sys
from uuid import UUID
from contextlib import asynccontextmanager

import asyncpg
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from urllib.parse import urlparse

from actr_scoring import ACTRConfig, score_and_rank_memories
from forgetting import run_forgetting_cycle
from hybrid_search import build_search_queries, reciprocal_rank_fusion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("claude-memory")

# Load config from .env next to this file, then from project root
_src_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_src_dir)
load_dotenv(os.path.join(_project_dir, ".env"))

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DATABASE = os.getenv("PG_DATABASE", "claude_memory")
PG_USER = os.getenv("PG_USER", os.environ.get("USER", ""))
PG_PASSWORD = os.getenv("PG_PASSWORD", "")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_parsed = urlparse(OLLAMA_HOST)
if _parsed.hostname not in ("localhost", "127.0.0.1"):
    raise RuntimeError(f"OLLAMA_HOST must be localhost/127.0.0.1. Got: {_parsed.hostname}")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

VALID_CATEGORIES = frozenset({
    "bugfix", "decision", "feature", "discovery", "refactor",
    "change", "learning", "pattern", "error_solution", "preference",
})

USER_ID = os.getenv("USER_ID", "default")
actr_config = ACTRConfig.from_env()

pool = None


async def get_pool():
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(
            host=PG_HOST,
            port=PG_PORT,
            database=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD or None,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
    return pool


@asynccontextmanager
async def lifespan(server):
    logger.info("Server starting")
    try:
        yield
    finally:
        global pool
        if pool is not None:
            logger.info("Closing connection pool")
            await pool.close()
            pool = None


mcp = FastMCP("claude-memory", lifespan=lifespan)


async def get_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]


def format_embedding(embedding: list[float]) -> str:
    return "[" + ",".join(str(x) for x in embedding) + "]"


@mcp.tool()
async def store_memory(
    content: str,
    category: str,
    summary: str = None,
    tags: list[str] = None,
    importance: float = 0.5,
    project: str = None,
) -> str:
    """Store a memory (learning, pattern, decision, error fix, etc).

    Args:
        content: Full memory content
        category: Type — bugfix, decision, feature, discovery, refactor, change, pattern, preference, learning, error_solution
        summary: Short summary (auto-generated if omitted)
        tags: Tags for filtering
        importance: Score 0.0-1.0
        project: Project context (optional)
    """
    if category not in VALID_CATEGORIES:
        return f"Invalid category '{category}'. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}"
    importance = max(0.0, min(1.0, importance))
    if len(content) > 50000:
        return "Content too long (max 50000 characters)."

    try:
        embedding = await get_embedding(content)
        if not summary:
            summary = content[:150] + "..." if len(content) > 150 else content

        db = await get_pool()
        async with db.acquire() as conn:
            result = await conn.fetchrow("""
                INSERT INTO memories
                (content, summary, category, tags, embedding, importance_score, project_context, user_id)
                VALUES ($1, $2, $3, $4, $5::vector, $6, $7, $8)
                RETURNING id
            """, content, summary, category, tags or [], format_embedding(embedding), importance, project, USER_ID)

        return f"Memory stored with ID: {result['id']}"
    except Exception as e:
        logger.error(f"store_memory failed: {e}", exc_info=True)
        return f"Error storing memory: {e}"


@mcp.tool()
async def retrieve_memories(
    query: str,
    max_results: int = 5,
    category: str = None,
    min_similarity: float = 0.3,
    tags: list[str] = None,
    project: str = None,
    include_forgotten: bool = False,
) -> str:
    """Search memories semantically (hybrid vector + trigram search with ACT-R scoring).

    Args:
        query: Search query
        max_results: Max results (default 5)
        category: Filter by category
        min_similarity: Minimum similarity 0.0-1.0 (default 0.3)
        tags: Filter by tags
        project: Filter by project
        include_forgotten: Include forgotten memories (default false)
    """
    max_results = max(1, min(100, max_results))
    min_similarity = max(0.0, min(1.0, min_similarity))

    try:
        query_embedding = await get_embedding(query)
        embedding_str = format_embedding(query_embedding)

        db = await get_pool()
        async with db.acquire() as conn:
            prefetch = actr_config.prefetch_limit if actr_config.use_actr else max_results

            vec_sql, vec_params, trgm_sql, trgm_params = build_search_queries(
                embedding_str=embedding_str,
                query_text=query,
                min_similarity=min_similarity,
                prefetch=prefetch,
                category=category,
                tags=tags,
                project=project,
                include_forgotten=include_forgotten,
                user_id=USER_ID,
            )

            vec_rows = await conn.fetch(vec_sql, *vec_params)

            trgm_rows = []
            try:
                trgm_rows = await conn.fetch(trgm_sql, *trgm_params)
            except Exception as trgm_err:
                logger.warning(f"Trigram search unavailable: {trgm_err}")

            vec_dicts = [dict(r) for r in vec_rows]
            trgm_dicts = [dict(r) for r in trgm_rows]
            fused = reciprocal_rank_fusion(vec_dicts, trgm_dicts)

            if not fused:
                return "No relevant memories found."

            if actr_config.use_actr:
                tag_counts = await conn.fetch("""
                    SELECT unnest(tags) as tag, COUNT(*) as cnt
                    FROM memories
                    WHERE tags IS NOT NULL AND array_length(tags, 1) > 0
                    GROUP BY tag
                """)
                tag_fan = {r["tag"].lower(): r["cnt"] for r in tag_counts}

                scored = score_and_rank_memories(
                    rows=fused,
                    query_tags=tags,
                    tag_fan_counts=tag_fan,
                    config=actr_config,
                    query=query,
                    category=category,
                )
                final_rows = scored[:max_results]
            else:
                fused.sort(key=lambda r: r["sim"] * r["importance_score"], reverse=True)
                final_rows = fused[:max_results]

            # Update access timestamps
            ids = [row["id"] for row in final_rows]
            await conn.execute("""
                UPDATE memories
                SET last_accessed_at = NOW(),
                    access_count = access_count + 1,
                    access_timestamps = (
                        CASE
                            WHEN array_length(COALESCE(access_timestamps, '{}'), 1) >= 1000
                            THEN array_append(access_timestamps[2:], NOW())
                            ELSE array_append(COALESCE(access_timestamps, '{}'), NOW())
                        END
                    )
                WHERE id = ANY($1) AND (user_id = $2 OR user_id IS NULL)
            """, ids, USER_ID)

        results = []
        for row in final_rows:
            activation_info = ""
            if actr_config.use_actr and "activation_score" in row:
                activation_info = f", activation: {row['activation_score']:.2f}"
            rrf_info = ""
            if "rrf_score" in row:
                rrf_info = f", rrf: {row['rrf_score']:.4f}"
            project_info = ""
            if row.get("project_context"):
                project_info = f"\nProject: {row['project_context']}"
            results.append(f"""
---
**[{row['category']}]** (similarity: {row['sim']:.2f}, importance: {row['importance_score']:.1f}{activation_info}{rrf_info})
{row['content']}
Tags: {', '.join(row['tags']) if row['tags'] else 'none'}{project_info}
""")

        return f"## {len(final_rows)} memory(ies) found:\n" + "\n".join(results)

    except Exception as e:
        logger.error(f"retrieve_memories failed: {e}", exc_info=True)
        return f"Error retrieving memories: {e}"


@mcp.tool()
async def list_memories(
    limit: int = 20,
    category: str = None,
    tags: list[str] = None,
    project: str = None,
) -> str:
    """List recent memories.

    Args:
        limit: Number of memories (default 20)
        category: Filter by category
        tags: Filter by tags
        project: Filter by project
    """
    limit = max(1, min(100, limit))
    if category and category not in VALID_CATEGORIES:
        return f"Invalid category '{category}'. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}"

    try:
        db = await get_pool()
        async with db.acquire() as conn:
            conditions = ["(user_id = $1 OR user_id IS NULL)"]
            params: list = [USER_ID]
            idx = 2

            if category:
                conditions.append(f"category = ${idx}")
                params.append(category)
                idx += 1
            if tags:
                conditions.append(f"tags @> ${idx}")
                params.append(tags)
                idx += 1
            if project:
                conditions.append(f"project_context = ${idx}")
                params.append(project)
                idx += 1

            where = "WHERE " + " AND ".join(conditions)
            params.append(limit)

            rows = await conn.fetch(f"""
                SELECT id, summary, category, tags, importance_score,
                       created_at, access_count, project_context, memory_status
                FROM memories
                {where}
                ORDER BY created_at DESC
                LIMIT ${idx}
            """, *params)

        if not rows:
            return "No memories stored."

        results = []
        for row in rows:
            project_info = f" | project: {row['project_context']}" if row.get("project_context") else ""
            status = f" [{row['memory_status']}]" if row.get("memory_status") and row["memory_status"] != "active" else ""
            results.append(
                f"- **{row['category']}**{status} | {row['summary'][:80]}... | "
                f"importance: {row['importance_score']:.1f} | accessed: {row['access_count']}x{project_info}"
            )

        return f"## {len(rows)} memory(ies):\n" + "\n".join(results)

    except Exception as e:
        logger.error(f"list_memories failed: {e}", exc_info=True)
        return f"Error listing memories: {e}"


@mcp.tool()
async def delete_memory(memory_id: str) -> str:
    """Delete a memory by ID.

    Args:
        memory_id: UUID of the memory to delete
    """
    try:
        parsed_id = UUID(memory_id)
    except (ValueError, AttributeError):
        return "Invalid memory ID format. Must be a valid UUID."

    try:
        db = await get_pool()
        async with db.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memories WHERE id = $1 AND (user_id = $2 OR user_id IS NULL)",
                parsed_id, USER_ID,
            )
        if result == "DELETE 1":
            return f"Memory {memory_id} deleted."
        return f"Memory {memory_id} not found."
    except Exception as e:
        logger.error(f"delete_memory failed: {e}", exc_info=True)
        return f"Error deleting memory: {e}"


@mcp.tool()
async def memory_stats() -> str:
    """Show memory database statistics including ACT-R status breakdown."""
    try:
        db = await get_pool()
        async with db.acquire() as conn:
            uf = "(user_id = $1 OR user_id IS NULL)"
            total = await conn.fetchval(f"SELECT COUNT(*) FROM memories WHERE {uf}", USER_ID)
            by_category = await conn.fetch(f"""
                SELECT category, COUNT(*) as count
                FROM memories WHERE {uf}
                GROUP BY category ORDER BY count DESC
            """, USER_ID)
            recent = await conn.fetchval(f"""
                SELECT COUNT(*) FROM memories
                WHERE created_at > NOW() - INTERVAL '7 days' AND {uf}
            """, USER_ID)
            most_accessed = await conn.fetch(f"""
                SELECT summary, access_count FROM memories
                WHERE {uf} ORDER BY access_count DESC LIMIT 5
            """, USER_ID)
            by_status = await conn.fetch(f"""
                SELECT COALESCE(memory_status, 'active') as status, COUNT(*) as count
                FROM memories WHERE {uf}
                GROUP BY COALESCE(memory_status, 'active') ORDER BY count DESC
            """, USER_ID)
            avg_activation = await conn.fetchval(f"""
                SELECT AVG(actr_activation) FROM memories
                WHERE actr_activation IS NOT NULL AND {uf}
            """, USER_ID)

        stats = f"""## Memory Statistics

**Total**: {total} memories
**This week**: {recent} new
**Scoring**: {'ACT-R cognitive' if actr_config.use_actr else 'Cosine classic'}

### By category:
"""
        for row in by_category:
            stats += f"- {row['category']}: {row['count']}\n"

        stats += "\n### By status:\n"
        for row in by_status:
            stats += f"- {row['status']}: {row['count']}\n"

        if avg_activation is not None:
            stats += f"\n**Average activation**: {avg_activation:.2f}\n"

        stats += "\n### Most accessed:\n"
        for row in most_accessed:
            if row["summary"]:
                stats += f"- ({row['access_count']}x) {row['summary'][:60]}...\n"

        return stats

    except Exception as e:
        logger.error(f"memory_stats failed: {e}", exc_info=True)
        return f"Error getting stats: {e}"


@mcp.tool()
async def memory_forgetting_cycle() -> str:
    """Run ACT-R strategic forgetting cycle.

    Recalculates activation for all memories and updates their status:
    active (A>0), dormant (-2<A<=0), forgotten (A<=-2).
    Forgotten memories remain in database but are excluded from default results.
    """
    try:
        db = await get_pool()
        result = await run_forgetting_cycle(db, actr_config)
        return result
    except Exception as e:
        logger.error(f"forgetting_cycle failed: {e}", exc_info=True)
        return f"Error running forgetting cycle: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
