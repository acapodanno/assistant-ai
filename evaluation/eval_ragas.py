#!/usr/bin/env python3
import json
import os
from loguru import logger
from datasets import Dataset
from src.rag import run_rag, run_ingestion

try:
    import sys
    import types
    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
    class _ChatVertexAI: pass
    _vertexai_stub.ChatVertexAI = _ChatVertexAI
    sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertexai_stub)

    from ragas import evaluate as ragas_evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics.collections import answer_relevancy, faithfulness, context_precision, context_recall
    RAGAS_AVAILABLE = True
except Exception as e:
    logger.warning(f"RAGAS non disponibile: {e}")
    RAGAS_AVAILABLE = False

def run_ragas():
    if not RAGAS_AVAILABLE: return
    run_ingestion()
    with open("evaluation/test_dataset_rag.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    ragas_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in eval_data:
        query = item["query"]
        res = run_rag(query)
        ragas_data["question"].append(query)
        ragas_data["answer"].append(res.get("answer", ""))
        ragas_data["contexts"].append([doc.page_content for doc in res.get("context", [])])
        ragas_data["ground_truth"].append(item["ground_truth"])

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    rl = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    re = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    res_eval = ragas_evaluate(
        Dataset.from_dict(ragas_data),
        metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
        llm=rl, embeddings=re
    )
    logger.info(f"Risultati Ragas: {res_eval}")

if __name__ == "__main__":
    run_ragas()
