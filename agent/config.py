from langchain_community.embeddings import JinaEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = JinaEmbeddings(
        model_name="jina-embeddings-v2-base-code",
        jina_api_key=os.getenv("JINA_API_KEY")
)