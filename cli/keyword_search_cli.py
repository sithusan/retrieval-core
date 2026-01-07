#!/usr/bin/env python3

import argparse
from lib.keyword_search import build_command, search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build movies to Inverted Index")

    args = parser.parse_args()

    match args.command:
        case "build":
            build_command()
        case "search":
            search_command(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
