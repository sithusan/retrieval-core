import argparse
from lib.augmented_generation import rag
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
            rag(args.query, RRF_WEIGHT, SEARCH_LIMIT)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
