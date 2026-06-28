#!/usr/bin/env python3
import json
from loguru import logger
from src.rag.run_ingestion import run_ingestion
from src.agent.react_agent import GreenThumbAgent
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate as deepeval_evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig


def load_dataset(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def run_deepeval():
    logger.info("Initializing RAG database for agent evaluation...")
    run_ingestion()

    dataset_path = "evaluation/test_dataset_agent.json"
    logger.info(f"Loading agent dataset from {dataset_path}")
    eval_data = load_dataset(dataset_path)

    agent = GreenThumbAgent()
    test_cases = []

    logger.info("Running Agent to generate responses...")
    for item in eval_data:
        query = item["query"]
        ground_truth = item["ground_truth"]

        logger.info(f"Agent processing: '{query}'")
        raw_response = agent.run(query)
        try:
            response_obj = json.loads(raw_response)
            actual_output = response_obj.get("answer", raw_response)
        except (json.JSONDecodeError, TypeError):
            actual_output = str(raw_response)

        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=ground_truth,
        )
        test_cases.append(test_case)

    logger.info("Configuring DeepEval Metrics for Agent evaluation...")

    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.5,
        model="gpt-4o",
        include_reason=True,
    )

    correctness = GEval(
        name="Agent Correctness",
        criteria=(
            "Valuta se la risposta dell'agente è corretta e completa rispetto all'output atteso. "
            "Considera: (1) ha usato le informazioni giuste (ordine, policy)? "
            "(2) ha aperto un ticket quando necessario? "
            "(3) ha communicato chiaramente la situazione al cliente? "
            "(4) il tono è cordiale e professionale?"
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
        model="gpt-4o",
    )

    metrics = [answer_relevancy, correctness]

    logger.info("Starting DeepEval Agent Evaluation...")
    deepeval_evaluate(
        test_cases,
        metrics,
        async_config=AsyncConfig(run_async=True, max_concurrent=1, throttle_value=10),
        cache_config=CacheConfig(write_cache=True, use_cache=True),
    )
    logger.info("DeepEval Agent Evaluation Complete!")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_deepeval()
