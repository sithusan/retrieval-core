#!/usr/bin/env python3

import argparse
from lib.keyword_search import (
    build_command,
    search_command,
    tf_command,
    idf_command,
    tf_idf_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build movies to Inverted Index")

    tf_parser = subparsers.add_parser("tf", help="Get Term Frequency of given term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparsers.add_parser("idf", help="Inverse Document Frequency")
    idf_parser.add_argument("term", type=str, help="Term")

    tf_idf_parser = subparsers.add_parser(
        "tfidf", help="Get Term Frequencey and inverse term frequency"
    )
    tf_idf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_idf_parser.add_argument("term", type=str, help="Term")
    args = parser.parse_args()

    match args.command:
        case "build":
            build_command()
        case "search":
            search_command(args.query)
        case "tf":
            tf_command(args.doc_id, args.term)
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tf_idf_command(args.doc_id, args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
