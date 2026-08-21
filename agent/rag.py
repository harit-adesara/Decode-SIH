import os
from langchain_classic.retrievers.document_compressors.cohere_rerank import CohereRerank
from qdrant import vectorstore
from dotenv import load_dotenv

load_dotenv()

reranker = CohereRerank(cohere_api_key=os.getenv("COHERE_API_KEY"), top_n=5, model="rerank-v3.5")


def retrieve_and_rerank(query: str):

    docs = vectorstore.similarity_search(
        query,
        k=10
    )

    reranked_docs = reranker.compress_documents(docs, query)

    context = "\n\n".join(
        doc.page_content
        for doc in reranked_docs
    )

    return context