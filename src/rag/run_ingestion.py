from src.rag.ingestion import retrieve_documents, chunk_documents
from src.rag.retriever import build_vector_store, build_sparse_retriever
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

def run_ingestion():
    logger.info("Loading documents...")
    docs = retrieve_documents()
    logger.info(f"-> {len(docs)} documents loaded.")
    chunks = chunk_documents(docs)
    logger.info(f"-> {len(chunks)} chunks.")
    build_vector_store(chunks)
    build_sparse_retriever(chunks)
    logger.info("Ingestion completed!")

if __name__ == "__main__":
    run_ingestion()
