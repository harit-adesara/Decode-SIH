import os
import logging
from dotenv import load_dotenv
import cohere
from .qdrant import vectorstore

load_dotenv()

logger = logging.getLogger("bharatswasthya.rag")

_cohere_key = os.getenv("COHERE_API_KEY")
cohere_client = cohere.ClientV2(api_key=_cohere_key) if _cohere_key else None


def retrieve_and_rerank(query: str) -> str:
    """
    Retrieve top matching health scheme documents from Qdrant and rerank them using Cohere.
    Falls back gracefully to similarity search results if Cohere reranking encounters an issue.
    """
    try:
        docs = vectorstore.similarity_search(query, k=10)
        docs = [
            doc for doc in docs
            if doc.page_content and doc.page_content.strip()
        ]

        if not docs:
            return "No relevant government scheme information found."

        doc_texts = [doc.page_content for doc in docs]

        if cohere_client:
            try:
                rerank_response = cohere_client.rerank(
                    model="rerank-v3.5",
                    query=query,
                    documents=doc_texts,
                    top_n=min(5, len(doc_texts)),
                )
                selected_texts = [
                    doc_texts[item.index]
                    for item in rerank_response.results
                ]
            except Exception as rerank_err:
                logger.warning("Cohere rerank failed, falling back to vector search: %s", rerank_err)
                selected_texts = doc_texts[:5]
        else:
            selected_texts = doc_texts[:5]

        context = "\n\n".join(selected_texts).strip()
        return context or "No relevant government scheme information found."

    except Exception as e:
        logger.error("RAG retrieval failed: %s", e)
        return "No relevant government scheme information found."