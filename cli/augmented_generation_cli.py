import argparse
from lib.augmented_generation import rag, summarize
from lib.constants import RRF_WEIGHT, SEARCH_LIMIT


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="summarize the answser of the found movies"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for summarize")
    summarize_parser.add_argument(
        "--limit", type=int, default=SEARCH_LIMIT, help="Search query Limit"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            rag(args.query, RRF_WEIGHT, SEARCH_LIMIT)
        case "summarize":
            summarize(args.query, RRF_WEIGHT, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
