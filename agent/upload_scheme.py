import os
import re
from pathlib import Path
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

FILE_PATH = Path(__file__).resolve().parent / "Indian_Government_Health_Schemes_Complete.txt"
COLLECTION_NAME = "health_schemes"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")
EMBEDDING_SIZE = 1024


def get_qdrant_client() -> QdrantClient:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY is missing from .env")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def create_collection(client: QdrantClient):
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
    )


def divide_schemes():
    if not FILE_PATH.exists():
        raise FileNotFoundError(f"File not found: {FILE_PATH}")

    text = FILE_PATH.read_text(encoding="utf-8")
    pattern = re.compile(r"<<<SCHEME_START:(\d+)>>>\s*(.*?)\s*<<<SCHEME_END:\1>>>", re.DOTALL)
    matches = pattern.findall(text)

    schemes = []
    for scheme_id, content in matches:
        scheme_id = int(scheme_id)
        content = content.strip()
        name_match = re.search(r"(?m)^\s*\d+\.\s*(.+)$", content)
        scheme_name = " ".join(name_match.group(1).split()) if name_match else f"Scheme {scheme_id}"
        schemes.append({
            "id": scheme_id,
            "name": scheme_name,
            "text": content,
        })
    return schemes


def create_embedding(text: str):
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY is missing from .env")
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "jina-embeddings-v3",
        "input": [text],
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    embedding = data["data"][0]["embedding"]

    if len(embedding) != EMBEDDING_SIZE:
        raise ValueError(f"Embedding dimension mismatch: expected {EMBEDDING_SIZE}, got {len(embedding)}")
    return embedding


def upload_schemes(client: QdrantClient, schemes: list):
    points = []
    for scheme in schemes:
        embedding = create_embedding(scheme["text"])
        point = PointStruct(
            id=scheme["id"],
            vector=embedding,
            payload={
                "scheme_id": scheme["id"],
                "scheme_name": scheme["name"],
                "text": scheme["text"],
            },
        )
        points.append(point)

    client.upsert(collection_name=COLLECTION_NAME, points=points)


def main():
    client = get_qdrant_client()
    schemes = divide_schemes()
    if len(schemes) != 28:
        raise ValueError(f"Expected 28 schemes, found {len(schemes)}")
    create_collection(client)
    upload_schemes(client, schemes)


if __name__ == "__main__":
    main()