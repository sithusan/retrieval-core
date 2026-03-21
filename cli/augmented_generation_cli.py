import argparse
from lib.hybrid_search import rrf_search_with_reranking, result_from_llm
from lib.prompts import search_answer_prompt
from lib.constants import RRF_WEIGHT, SEARCH_LIMIT


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            result = rrf_search_with_reranking(
                query, RRF_WEIGHT, SEARCH_LIMIT, None, None
            )
            prompt = search_answer_prompt(query, result)
            rag_answer = result_from_llm(prompt)

            print("Search Results")
            for _, v in result.items():
                print(f"-{v['title']}")
            print(rag_answer)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
