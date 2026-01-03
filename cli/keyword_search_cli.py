#!/usr/bin/env python3

import argparse
import json
import os
import string


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

    processedQuery = processText(query)

    for movie in movies:
        processedMovieTitle = processText(movie["title"])
        if isMatch(processedQuery, processedMovieTitle):
            foundmovies.append(movie)
            print(f"{len(foundmovies)}. {movie["title"]}")

        if len(foundmovies) == limit:
            break


def loadMovies() -> dict:
    abs_path = os.path.abspath("./data/movies.json")
    movies_path = os.path.normpath(abs_path)

    file = open(movies_path, "r")

    return json.load(file)["movies"]


def processText(text: str) -> string:
    lowered = text.lower()
    punctuationRemoved = removePunctuation(lowered)
    tokenizated = tokenize(punctuationRemoved)

    return tokenizated


def removePunctuation(text: str) -> string:
    trans = {}
    for punctuation in string.punctuation:
        trans[punctuation] = ""

    return text.translate(str.maketrans(trans))


def tokenize(text: str) -> set:
    splitted = text.split(" ")

    return set(filter(None, splitted))


def isMatch(query_tokens: set, target_tokens: set) -> bool:
    for query_token in query_tokens:
        for target_token in target_tokens:
            if query_token in target_token:
                return True

    return False


if __name__ == "__main__":
    main()
