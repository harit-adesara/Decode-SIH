import os

from dotenv import load_dotenv
from langchain_classic.retrievers.document_compressors.cohere_rerank import CohereRerank

from .qdrant import vectorstore

load_dotenv()


reranker = CohereRerank(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    top_n=5,
    model="rerank-v3.5",
)


def retrieve_and_rerank(query: str):

    docs = vectorstore.similarity_search(
        query,
        k=10,
    )

    docs = [
        doc
        for doc in docs
        if doc.page_content
        and doc.page_content.strip()
    ]

    if not docs:
        return "No relevant government scheme information found."

    reranked_docs = reranker.compress_documents(
        docs,
        query,
    )

    context = "\n\n".join(
        doc.page_content
        for doc in reranked_docs
        if doc.page_content
        and doc.page_content.strip()
    )

    if not context:
        return "No relevant government scheme information found."

    return context