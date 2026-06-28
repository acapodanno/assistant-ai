from src.rag.ingestion import retrieve_documents, chunk_documents
from src.rag.retriever import build_vector_store, build_sparse_retriever
from dotenv import load_dotenv

load_dotenv()

from loguru import logger

def run_ingestion():
    logger.info("Loading documents from Knowledge Base...")
    docs = retrieve_documents()
    logger.info(f"-> {len(docs)} documents loaded.")

    logger.info("Chunking documents...")
    chunks = chunk_documents(docs)
    logger.info(f"-> {len(chunks)} chunks generated.")

    logger.info("Generating embeddings and populating ChromaDB...")
    build_vector_store(chunks)
    logger.info("ChromaDB populated successfully in ./chroma_db")

    logger.info("Generating BM25 index...")
    build_sparse_retriever(chunks)
    logger.info("BM25 index saved successfully in ./chroma_db/bm25_index.pkl")

    logger.info("Ingestion completed! RAG system is ready.")