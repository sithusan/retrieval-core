#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search,
)
from lib.constants import SEARCH_LIMIT


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the model to use the semantic search")
    subparsers.add_parser(
        "verify_embeddings", help="Verify the embeddings to use the semantic search"
    )

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed the text")
    embed_text_parser.add_argument("text", type=str, help="Text to be embedded")

    embed_query_text_parser = subparsers.add_parser(
        "embedquery", help="Embed the query"
    )
    embed_query_text_parser.add_argument("query", type=str, help="Query to be embedded")

    search_parser = subparsers.add_parser("search", help="Search Query")
    search_parser.add_argument("query", type=str, help="Query for search")
    search_parser.add_argument(
        "--limit", type=int, default=SEARCH_LIMIT, help="Search query Limit"
    )
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
