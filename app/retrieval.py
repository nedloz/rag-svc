# rag_service/retrieval.py
"""
Vector Retrieval through library schema
=====================================
Retrieves chunks using library.chunk_embeddings, library.chunks, and library.documents.
Applies document status and user profile (scope_json) filtering.
"""
import logging
from typing import List, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import os

logger = logging.getLogger("Retrieval")

TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", 5))


async def vector_search(
    session: AsyncSession,
    query_embedding: List[float],
    user_profile: dict,
    top_k: int = TOP_K_DEFAULT,
) -> List[dict]:
    """
    ANN-search in pgvector with document scope and status filtering.

    Parameters
    ----------
    session         : SQLAlchemy AsyncSession
    query_embedding : Query vector (matching indexed dimension, e.g. 1536)
    user_profile    : User profile dict with university_id, campus_id, etc.
    top_k           : limit of chunks to return (default=5)

    Returns
    -------
    List of dicts:
        {
            "chunk_id"   : str (UUID),
            "text"       : str,
            "similarity" : float,
            "metadata"   : dict,
            "doc_id"     : str (UUID),
            "doc_title"  : str
        }
    """
    # SQL logic for scope_json:
    # 1. Option is missing or empty array -> access allowed.
    # 2. Option is present -> user profile must contain a value present in the scope array.
    sql = text("""
        SELECT
            c.id::text                        AS chunk_id,
            c.text                            AS text,
            c.metadata_json                   AS metadata,
            d.id::text                        AS doc_id,
            d.title                           AS doc_title,
            1 - (e.embedding <=> :vec_vector) AS similarity
        FROM library.chunk_embeddings e
        JOIN library.chunks           c ON c.id = e.chunk_id
        JOIN library.documents        d ON d.id = c.document_id
        WHERE d.status = 'published'
        AND d.ingest_status = 'indexed'
        AND d.indexed_at IS NOT NULL

        AND (
            d.scope_json->'university_ids' IS NULL
            OR jsonb_array_length(d.scope_json->'university_ids') = 0
            OR (
                CAST(:university_id AS text) IS NOT NULL
                AND d.scope_json->'university_ids' @> jsonb_build_array(CAST(:university_id AS text))
            )
        )

        AND (
            d.scope_json->'campus_ids' IS NULL
            OR jsonb_array_length(d.scope_json->'campus_ids') = 0
            OR (
                CAST(:campus_id AS text) IS NOT NULL
                AND d.scope_json->'campus_ids' @> jsonb_build_array(CAST(:campus_id AS text))
            )
        )

        AND (
            d.scope_json->'faculty_ids' IS NULL
            OR jsonb_array_length(d.scope_json->'faculty_ids') = 0
            OR (
                CAST(:faculty_id AS text) IS NOT NULL
                AND d.scope_json->'faculty_ids' @> jsonb_build_array(CAST(:faculty_id AS text))
            )
        )

        AND (
            d.scope_json->'program_ids' IS NULL
            OR jsonb_array_length(d.scope_json->'program_ids') = 0
            OR (
                CAST(:program_id AS text) IS NOT NULL
                AND d.scope_json->'program_ids' @> jsonb_build_array(CAST(:program_id AS text))
            )
        )

        AND (
            d.scope_json->'years' IS NULL
            OR jsonb_array_length(d.scope_json->'years') = 0
            OR (
                CAST(:study_year AS int) IS NOT NULL
                AND d.scope_json->'years' @> jsonb_build_array(CAST(:study_year AS int))
            )
        )

        AND (
            d.scope_json->'roles' IS NULL
            OR jsonb_array_length(d.scope_json->'roles') = 0
            OR (
                CAST(:role AS text) IS NOT NULL
                AND d.scope_json->'roles' @> jsonb_build_array(CAST(:role AS text))
            )
        )
        ORDER BY e.embedding <=> :vec_vector
        LIMIT :k
    """)

    # Vector formatting for pgvector (string format)
    vec_str = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"

    params = {
        "vec_vector": vec_str,
        "k": top_k,
        "university_id": user_profile.get("university_id"),
        "campus_id": user_profile.get("campus_id"),
        "faculty_id": user_profile.get("faculty_id"),
        "program_id": user_profile.get("program_id"),
        "study_year": user_profile.get("year"),
        "role": user_profile.get("role"),
    }

    try:
        result = await session.execute(sql, params)
        rows = result.mappings().all()
    except Exception as e:
        logger.error(f"SQL Execution error in retrieval: {e}")
        return []

    return [
        {
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "similarity": float(row["similarity"]),
            "metadata": row["metadata"] or {},
            "doc_id": row["doc_id"],
            "doc_title": row["doc_title"],
        }
        for row in rows
    ]

async def get_chunks_by_ids(session: AsyncSession, chunk_ids: List[str]) -> List[dict]:
    """
    Fetch chunks by their UUIDs.

    Parameters
    ----------
    session   : SQLAlchemy AsyncSession
    chunk_ids : List of UUID strings

    Returns
    -------
    List of dicts with chunk text and document title.
    """
    if not chunk_ids:
        return []

    sql = text("""
        SELECT
            c.id::text           AS chunk_id,
            c.text               AS text,
            c.metadata_json      AS metadata,
            d.id::text           AS doc_id,
            d.title              AS doc_title
        FROM library.chunks c
        JOIN library.documents d ON d.id = c.document_id
        WHERE c.id = ANY(:ids::uuid[])
    """)

    try:
        result = await session.execute(sql, {"ids": chunk_ids})
        rows = result.mappings().all()
    except Exception as e:
        logger.error(f"SQL Execution error in get_chunks_by_ids: {e}")
        return []

    return [
        {
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "metadata": row["metadata"] or {},
            "doc_id": row["doc_id"],
            "doc_title": row["doc_title"],
        }
        for row in rows
    ]
