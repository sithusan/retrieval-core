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
        retrieved = rrf_search_with_reranking(v["query"], 60, limit, None, None)
        relevent_retrieved = {}

        for k, item in retrieved.items():
            if item["title"] in v["relevant_docs"]:
                relevent_retrieved[k] = v

        precision = len(relevent_retrieved) / len(retrieved)
        recall = len(relevent_retrieved) / len(v["relevant_docs"])
        f1 = 2 * (precision * recall) / (precision + recall)

        retrieved_titles = ", ".join(doc["title"] for doc in retrieved.values())
        relevant_titles = ", ".join(v["relevant_docs"])

        print(f"- Query: {v['query']}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {retrieved_titles}")
        print(f"  - Relevant: {relevant_titles}")


if __name__ == "__main__":
    main()
