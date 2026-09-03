import json
import os
from deepeval.metrics import ContextualRecallMetric,ContextualPrecisionMetric
from src.reranker import RerankingRetriever
from deepeval import evaluate
from deepeval.test_case import LLMTestCase 
from deepeval.models import OllamaModel
from deepeval.evaluate.configs import AsyncConfig
from dotenv import load_dotenv
load_dotenv()

GOLDEN_PATH = "goldens/retriever_golden.json"

JUDGE_MODELS = OllamaModel(
    model="gemma:2b",
    base_url="http://localhost:11434",
    temperature=0    
    )
THRESHOLD = 0.7

with open(GOLDEN_PATH) as f:
    golden = json.load(f)

retriever = RerankingRetriever()

test_cases = []

for g in golden:
    retrieved = retriever.invoke(g["query"])
    retrieval_context = [doc.page_content for doc in retrieved]
    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output="generator not evaluated"
        )
    )

metrics = [
    ContextualPrecisionMetric(threshold=THRESHOLD,model=JUDGE_MODELS),
    ContextualRecallMetric(threshold=THRESHOLD,model=JUDGE_MODELS)
]

config = AsyncConfig(
    max_concurrent=1
)

evaluate(test_cases = test_cases,metrics=metrics,async_config=config)
