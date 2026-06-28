from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
import pickle
import os

BM25_PATH = "./chroma_db/bm25_index.pkl"

def build_vector_store(chunked_documents: list[Document]):
    # Cancella la collection esistente per evitare accumulo di dati stantii/duplicati
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("greenthumb_kb")
    except Exception:
        pass  # collection non esisteva, va bene

    vector_store = Chroma.from_documents(
        documents=chunked_documents,
        embedding=OpenAIEmbeddings(
            model="text-embedding-3-large",
        ),
        persist_directory="./chroma_db",
        collection_name="greenthumb_kb",
    )
    return vector_store

def build_sparse_retriever(chunked_documents: list[Document]) -> BM25Retriever:
    """Costruisce e salva il retriever BM25 su disco."""
    retriever = BM25Retriever.from_documents(chunked_documents)
    retriever.k = 3
    # Assicurati che la cartella esista
    os.makedirs("./chroma_db", exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(retriever, f)
    return retriever

def get_dense_retriever():
    """Carica il retriever denso da ChromaDB."""
    vector_store = Chroma(
        persist_directory="./chroma_db",
        collection_name="greenthumb_kb",
        embedding_function=OpenAIEmbeddings(
             model="text-embedding-3-large",
        ),
    )
    return vector_store.as_retriever(
        search_type="mmr",  # Max Marginal Relevance
        search_kwargs={'k': 3, 'fetch_k': 10}
    )

def get_sparse_retriever() -> BM25Retriever:
    """Carica il retriever BM25 dal disco."""
    if not os.path.exists(BM25_PATH):
        raise FileNotFoundError(f"Indice BM25 non trovato in {BM25_PATH}. Eseguire l'ingestion.")
    with open(BM25_PATH, "rb") as f:
        retriever = pickle.load(f)
    return retriever

def get_hybrid_retriever(query: str, bm25_weight: float = 0.3) -> list[Document]:
    """
    Hybrid retrieval via Reciprocal Rank Fusion (RRF).
    Carica gli indici (BM25 e Dense) e li interroga senza ricalcolare.
    bm25_weight: peso del BM25 (0.3 = 30% sparse, 70% dense)
    """
    bm25 = get_sparse_retriever()
    dense = get_dense_retriever()

    bm25_results = bm25.invoke(query)
    dense_results = dense.invoke(query)

    # Reciprocal Rank Fusion
    scores: dict[str, float] = {}
    seen: dict[str, Document] = {}

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