#!/usr/bin/env python3

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            search(args.query)
            pass
        case _:
            parser.print_help()


def search(query: str) -> None:
    print(f"Searching for: {query}")

    limit = 5
    movies = loadMovies()
    foundmovies = []

    for movie in movies:
        if query in movie["title"]:
            foundmovies.append(movie)
            print(f"{len(foundmovies)}. {movie["title"]}")

        if len(foundmovies) == limit:
            break


def loadMovies() -> dict:
    abs_path = os.path.abspath("./data/movies.json")
    movies_path = os.path.normpath(abs_path)

    file = open(movies_path, "r")

    return json.load(file)["movies"]


if __name__ == "__main__":
    main()
