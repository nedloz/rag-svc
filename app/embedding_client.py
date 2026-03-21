import os
import httpx

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-svc:8000/embed")
SERVICE_NAME = os.getenv("SERVICE_NAME", "rag-svc")
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN", "")
INTERNAL_AUTH_HEADER_NAME = os.getenv("INTERNAL_AUTH_HEADER_NAME", "X-Service-Token")
INTERNAL_SERVICE_NAME_HEADER = os.getenv("INTERNAL_SERVICE_NAME_HEADER", "X-Service-Name")


def build_internal_auth_headers() -> dict[str, str]:
    return {
        INTERNAL_SERVICE_NAME_HEADER: SERVICE_NAME,
        INTERNAL_AUTH_HEADER_NAME: SERVICE_TOKEN,
    }


async def embed_one(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            EMBEDDING_SERVICE_URL,
            json={"texts": [text], "normalize": True},
            headers=build_internal_auth_headers(),
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"][0]