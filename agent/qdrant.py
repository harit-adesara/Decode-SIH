import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from .config import embeddings
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL:
    raise ValueError("QDRANT_URL is missing from .env")

if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY is missing from .env")


client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=120,
    prefer_grpc=False,
)


vectorstore = QdrantVectorStore(
    client=client,
    collection_name="health_schemes",
    embedding=embeddings,
    content_payload_key="text",
)