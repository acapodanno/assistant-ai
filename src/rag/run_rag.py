from src.rag.retriever import get_hybrid_retriever
from src.rag.rag_chain import setup_rag_chain
from dotenv import load_dotenv

load_dotenv()

from loguru import logger

def run_rag(query: str) -> dict:
    """
    Esegue il retrieval ibrido e la RAG chain per rispondere a una query.
    Assicura di aver eseguito 'python src/rag/run_ingestion.py' prima.
    """
    try:
        retrieved_docs = get_hybrid_retriever(query=query)
        rag_chain = setup_rag_chain()
        result = rag_chain({"input": query, "context": retrieved_docs})
        return result
    except FileNotFoundError as e:
        logger.error(f"RAG database is not initialized. ({str(e)})")
        return {"answer": f"Error: RAG database is not initialized. ({str(e)})", "context": []}

if __name__ == "__main__":
    question = "Quali sono le politiche di reso di GreenThumb?"
    logger.info(f"Test question: {question}")
    try:
        result = run_rag(question)
        logger.info(result.get("answer", ""))
    except Exception as e:
        logger.error(f"Error: {e}")