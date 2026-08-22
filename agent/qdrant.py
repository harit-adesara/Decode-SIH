from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from .config import embeddings
from qdrant_client.models import VectorParams, Distance
import os
from dotenv import load_dotenv

load_dotenv()


client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=120,
    prefer_grpc=True
)


if not client.collection_exists("github_rag"):
    client.create_collection(
        collection_name="github_rag",
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE
        )
    )


vectorstore = QdrantVectorStore(
    client=client,
    collection_name="github_rag",
    embedding=embeddings
)