"""
Hybrid Search: Vector + Trigram with Reciprocal Rank Fusion (RRF)

Combines pgvector cosine similarity with pg_trgm trigram matching,
then fuses results using RRF.
"""

import logging

logger = logging.getLogger("claude-memory")


def build_search_queries(
    embedding_str: str,
    query_text: str,
    min_similarity: float = 0.3,
    prefetch: int = 50,
    category: str | None = None,
    tags: list[str] | None = None,
    project: str | None = None,
    include_forgotten: bool = False,
    user_id: str | None = None,
) -> tuple[str, list, str, list]:
    """Build parallel SQL queries for vector and trigram search."""

    def _build_filters(param_offset: int, exclude_forgotten: bool) -> tuple[str, list]:
        clauses = []
        params = []

        if exclude_forgotten:
            clauses.append("(memory_status IS NULL OR memory_status != 'forgotten')")

        if user_id:
            params.append(user_id)
            clauses.append(f"(user_id = ${param_offset + len(params)} OR user_id IS NULL)")

        if category:
            params.append(category)
            clauses.append(f"category = ${param_offset + len(params)}")

        if tags:
            params.append(tags)
            clauses.append(f"tags @> ${param_offset + len(params)}")

        if project:
            params.append(project)
            clauses.append(f"project_context = ${param_offset + len(params)}")

        where = ""
        if clauses:
            where = "AND " + " AND ".join(clauses)
        return where, params

    # Vector query: $1=embedding, $2=min_sim, $3=limit.
    # Forgotten rows stay eligible here so a strongly-relevant memory can be rescued by
    # semantic similarity (the ACT-R tau threshold and min_similarity floor still gate it).
    # The caller promotes any surfaced forgotten row back out of 'forgotten' on return.
    vec_where, vec_filter_params = _build_filters(param_offset=3, exclude_forgotten=False)
    vector_sql = f"""
        SELECT
            id, content, summary, category, tags,
            importance_score, created_at,
            access_timestamps, memory_status,
            project_context,
            1 - (embedding <=> $1::vector) as sim
        FROM memories
        WHERE 1 - (embedding <=> $1::vector) >= $2
          {vec_where}
        ORDER BY (1 - (embedding <=> $1::vector)) DESC
        LIMIT $3
    """
    vector_params = [embedding_str, min_similarity, prefetch] + vec_filter_params

    # Trigram query: $1=query_text, $2=embedding, $3=limit.
    # Lexical overlap alone should not resurrect a forgotten memory, so this branch keeps
    # excluding them unless the caller explicitly asked to include forgotten rows.
    trgm_where, trgm_filter_params = _build_filters(
        param_offset=3, exclude_forgotten=not include_forgotten
    )
    trigram_sql = f"""
        SELECT
            id, content, summary, category, tags,
            importance_score, created_at,
            access_timestamps, memory_status,
            project_context,
            similarity(content || ' ' || COALESCE(summary, ''), $1) as trgm_sim,
            1 - (embedding <=> $2::vector) as sim
        FROM memories
        WHERE similarity(content || ' ' || COALESCE(summary, ''), $1) >= 0.05
          {trgm_where}
        ORDER BY similarity(content || ' ' || COALESCE(summary, ''), $1) DESC
        LIMIT $3
    """
    trigram_params = [query_text, embedding_str, prefetch] + trgm_filter_params

    return vector_sql, vector_params, trigram_sql, trigram_params


def reciprocal_rank_fusion(
    vector_results: list[dict],
    text_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Fuse two ranked lists using RRF. Vector results preferred for ties."""
    rrf_scores: dict[str, float] = {}
    doc_store: dict[str, dict] = {}

    for rank, row in enumerate(vector_results):
        doc_id = str(row["id"])
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        doc_store[doc_id] = row

    for rank, row in enumerate(text_results):
        doc_id = str(row["id"])
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in doc_store:
            doc_store[doc_id] = row

    fused = []
    for doc_id, score in rrf_scores.items():
        entry = dict(doc_store[doc_id])
        entry["rrf_score"] = score
        fused.append(entry)

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused
