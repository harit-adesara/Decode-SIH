import os
from dotenv import load_dotenv
from langchain_community.embeddings import JinaEmbeddings

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")

if not JINA_API_KEY:
    raise ValueError("JINA_API_KEY is missing from .env")

embeddings = JinaEmbeddings(
    model_name="jina-embeddings-v3",
    jina_api_key=JINA_API_KEY,
)