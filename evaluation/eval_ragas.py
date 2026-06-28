#!/usr/bin/env python3
import json
import os
from loguru import logger
from datasets import Dataset

# For RAG execution
from src.rag.run_rag import run_rag
from src.rag.run_ingestion import run_ingestion

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall
    )
    RAGAS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAGAS non è disponibile a causa di incompatibilità con le versioni recenti di langchain_community: {e}")
    RAGAS_AVAILABLE = False

def load_dataset(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def run_ragas():
    if not RAGAS_AVAILABLE:
        logger.error("Impossibile eseguire la valutazione RAGAS senza le dipendenze corrette.")
        return

    logger.info("Initializing RAG database for evaluation...")
    run_ingestion()

    dataset_path = "evaluation/test_dataset_rag.json"
    logger.info(f"Loading dataset from {dataset_path}")
    eval_data = load_dataset(dataset_path)

    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    logger.info("Running RAG pipeline to generate answers and contexts...")
    for item in eval_data:
        query = item["query"]
        ground_truth = item["ground_truth"]
        
        result = run_rag(query)
        answer = result.get("answer", "")
        context_docs = result.get("context", [])
        context_texts = [doc.page_content for doc in context_docs]

        ragas_data["question"].append(query)
        ragas_data["answer"].append(answer)
        ragas_data["contexts"].append(context_texts)
        ragas_data["ground_truth"].append(ground_truth)

    logger.info("Starting RAGAS Evaluation...")
    ragas_dataset = Dataset.from_dict(ragas_data)
    
    metrics = [
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall
    ]

    ragas_results = ragas_evaluate(ragas_dataset, metrics=metrics)
    logger.info(f"RAGAS Final Scores:\n{ragas_results}")
    
    df = ragas_results.to_pandas()
    df.to_csv("evaluation/ragas_results.csv", index=False)
    logger.info("Risultati dettagliati salvati in evaluation/ragas_results.csv")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_ragas()
