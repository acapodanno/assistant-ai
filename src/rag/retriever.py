from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
import pickle
import os

BM25_PATH = "./chroma_db/bm25_index.pkl"

def build_vector_store(chunked_documents: list[Document]):
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("greenthumb_kb")
    except Exception:
        pass

    vector_store = Chroma.from_documents(
        documents=chunked_documents,
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_directory="./chroma_db",
        collection_name="greenthumb_kb",
    )
    return vector_store

def build_sparse_retriever(chunked_documents: list[Document]) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(chunked_documents)
    retriever.k = 3
    os.makedirs("./chroma_db", exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(retriever, f)
    return retriever

def get_dense_retriever():
    vector_store = Chroma(
        persist_directory="./chroma_db",
        collection_name="greenthumb_kb",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    )
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 3, 'fetch_k': 10}
    )

def get_sparse_retriever() -> BM25Retriever:
    if not os.path.exists(BM25_PATH):
        raise FileNotFoundError(f"Indice BM25 non trovato in {BM25_PATH}. Eseguire l'ingestion.")
    with open(BM25_PATH, "rb") as f:
        retriever = pickle.load(f)
    return retriever

def get_hybrid_retriever(query: str, bm25_weight: float = 0.3) -> list[Document]:
    bm25 = get_sparse_retriever()
    dense = get_dense_retriever()
    bm25_results = bm25.invoke(query)
    dense_results = dense.invoke(query)

    scores = {}
    seen = {}
    for rank, doc in enumerate(bm25_results):
        key = doc.page_content[:120]
        scores[key] = scores.get(key, 0) + bm25_weight / (rank + 1)
        seen[key] = doc

    for rank, doc in enumerate(dense_results):
        key = doc.page_content[:120]
        scores[key] = scores.get(key, 0) + (1 - bm25_weight) / (rank + 1)
        seen[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [seen[k] for k, _ in ranked[:3]]
