#!/usr/bin/env python3
import json
from loguru import logger
from src.rag import run_ingestion
from src.agent import GreenThumbAgent
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate as deepeval_evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig

def run_deepeval():
    run_ingestion()
    with open("evaluation/test_dataset_agent.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    agent = GreenThumbAgent()
    test_cases = []
    for item in eval_data:
        query = item["query"]
        raw_res = agent.run(query)
        try:
            res_obj = json.loads(raw_res)
            act_out = res_obj.get("answer", raw_res)
        except Exception:
            act_out = str(raw_res)
        test_cases.append(LLMTestCase(input=query, actual_output=act_out, expected_output=item["ground_truth"]))

    answer_relevancy = AnswerRelevancyMetric(threshold=0.5, model="gpt-4o")
    correctness = GEval(
        name="Agent Correctness",
        criteria="Valuta se la risposta rispetta le indicazioni (strumenti usati, escalation, tono).",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5, model="gpt-4o"
    )
    deepeval_evaluate(test_cases, [answer_relevancy, correctness], async_config=AsyncConfig(run_async=True, max_concurrent=1, throttle_value=10))

if __name__ == "__main__":
    run_deepeval()
