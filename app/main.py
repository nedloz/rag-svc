"""
Retrieval Service
=================
1. Receives question + user profile from chat-svc.
2. Embeds the question.
3. Performs filtered vector search.
4. Sends retrieved chunks to llm-svc over HTTP.
5. Returns immediate ack to chat-svc.
"""
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from retrieval import vector_search, get_chunks_by_ids
from embedding_client import embed_one

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("RetrievalService")

DATABASE_URL = os.environ["DATABASE_URL"]
TOP_K_DEFAULT = int(os.environ.get("TOP_K_DEFAULT", "5"))
LLM_GENERATE_URL = os.environ.get("LLM_GENERATE_URL", "http://llm-svc:8000/generate")
SERVICE_NAME = os.environ.get("SERVICE_NAME", "rag-svc")
SERVICE_TOKEN = os.environ.get("SERVICE_TOKEN", "change-me")
INTERNAL_AUTH_ENABLED = os.environ.get("INTERNAL_AUTH_ENABLED", "false").lower() == "true"
TRUSTED_SERVICE_TOKENS_RAW = os.environ.get("TRUSTED_SERVICE_TOKENS", "{}")
INTERNAL_AUTH_HEADER_NAME = os.environ.get("INTERNAL_AUTH_HEADER_NAME", "X-Service-Token")
INTERNAL_SERVICE_NAME_HEADER = os.environ.get("INTERNAL_SERVICE_NAME_HEADER", "X-Service-Name")
EMBEDDER_NAME = os.environ.get("EMBEDDER_NAME", "mixedbread-ai/mxbai-embed-large-v1")
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)



app = FastAPI(title="HSE Retrieval Service", version="2.0.0")


class UserProfile(BaseModel):
    university_id: Optional[str] = None
    campus_id: Optional[str] = None
    faculty_id: Optional[str] = None
    program_id: Optional[str] = None
    year: Optional[int] = None
    group_name: Optional[str] = None
    role: Optional[str] = None


class RetrievalRequest(BaseModel):
    request_id: str
    message_id: str
    assistant_message_id: str
    session_id: str
    user_id: str
    question: str
    profile: UserProfile = Field(default_factory=UserProfile)
    top_k: int = TOP_K_DEFAULT


class RetrievalAcceptedResponse(BaseModel):
    status: str
    request_id: str
    assistant_message_id: str
    retrieved_count: int


class FetchChunksRequest(BaseModel):
    chunk_ids: List[str]


def _trusted_service_tokens() -> dict[str, str]:
    try:
        return json.loads(TRUSTED_SERVICE_TOKENS_RAW)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid TRUSTED_SERVICE_TOKENS JSON") from exc


def build_internal_auth_headers() -> dict[str, str]:
    return {INTERNAL_SERVICE_NAME_HEADER: SERVICE_NAME, INTERNAL_AUTH_HEADER_NAME: SERVICE_TOKEN}


def verify_internal_service_request(
    x_service_name: str = Header(..., alias=INTERNAL_SERVICE_NAME_HEADER),
    x_service_token: str = Header(..., alias=INTERNAL_AUTH_HEADER_NAME),
):
    if not INTERNAL_AUTH_ENABLED:
        return x_service_name
    expected_token = _trusted_service_tokens().get(x_service_name)
    if not expected_token or expected_token != x_service_token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid internal service token")
    return x_service_name


@app.get("/ping")
async def ping():
    return {
        "status": "ok",
        "service": "Retrieval Service",
        "embedder": EMBEDDER_NAME,
        "llm_generate_url": LLM_GENERATE_URL,
    }


@app.post("/retrieve", response_model=RetrievalAcceptedResponse)
async def retrieve(req: RetrievalRequest, _: str = Depends(verify_internal_service_request)):
    from embedding_client import embed_one
    started_at = time.perf_counter()
    try:
        query_vec = await embed_one(req.question)
    except Exception as exc:
        logger.error("Embedding error: %s", exc)
        raise HTTPException(500, f"Error generating embedding: {exc}")
    
    async with async_session() as session:
        try:
            chunks = await vector_search(
                session=session,
                query_embedding=query_vec,
                user_profile=req.profile.model_dump(),
                top_k=req.top_k,
            )
        except Exception as exc:
            logger.error("Vector search error: %s", exc)
            raise HTTPException(500, f"Database search error: {exc}")

    retrieval_ms = int((time.perf_counter() - started_at) * 1000)
    rag_result: dict[str, Any] = {
        "question": req.question,
        "chunks": chunks,
        "query_embedding_model": EMBEDDER_NAME,
        "filters_json": req.profile.model_dump(),
        "retrieved_chunk_ids": [item["chunk_id"] for item in chunks],
        "retrieved_doc_ids": list(dict.fromkeys(item["doc_id"] for item in chunks)),
        "scores_json": {item["chunk_id"]: item["similarity"] for item in chunks},
        "retrieval_ms": retrieval_ms,
    }

    llm_payload = {
        "request_id": req.request_id,
        "message_id": req.message_id,
        "assistant_message_id": req.assistant_message_id,
        "session_id": req.session_id,
        "user_id": req.user_id,
        "question": req.question,
        "profile": req.profile.model_dump(),
        "rag": rag_result,
    }
    headers = build_internal_auth_headers()
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(LLM_GENERATE_URL, json=llm_payload, headers=headers)
        response.raise_for_status()

    logger.info("Question accepted request_id=%s retrieved=%s", req.request_id, len(chunks))
    return RetrievalAcceptedResponse(
        status="accepted",
        request_id=req.request_id,
        assistant_message_id=req.assistant_message_id,
        retrieved_count=len(chunks),
    )


@app.post("/fetch_chunks")
async def fetch_chunks(req: FetchChunksRequest, _: str = Depends(verify_internal_service_request)):
    """
    Directly fetch chunks by their IDs.
    Used for generating summaries from history.
    """
    async with async_session() as session:
        try:
            chunks = await get_chunks_by_ids(session, req.chunk_ids)
            return {"chunks": chunks}
        except Exception as exc:
            logger.error("Fetch chunks error: %s", exc)
            raise HTTPException(500, f"Error fetching chunks: {exc}")
