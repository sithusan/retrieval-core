import argparse

from lib.constants import SEARCH_LIMIT
from lib.search_utils import get_path
from lib.hybrid_search import rrf_search_with_reranking
import json


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=SEARCH_LIMIT,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open(get_path("./data/golden_dataset.json"), "rb") as file:
        golden_dataset = json.load(file)

    for v in golden_dataset["test_cases"]:
        result = rrf_search_with_reranking(v["query"], 60, limit, None, None)
        precision = len(v["relevant_docs"]) / len(result)
        print(len(result), len(v["relevant_docs"]))
        recall = len(result) / len(v["relevant_docs"])

        retrieved_titles = ", ".join(doc["title"] for doc in result.values())
        relevant_titles = ", ".join(v["relevant_docs"])

        print(f"- Query: {v['query']}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - Retrieved: {retrieved_titles}")
        print(f"  - Relevant: {relevant_titles}")


if __name__ == "__main__":
    main()
