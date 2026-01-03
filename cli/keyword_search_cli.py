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

    processed_query = processText(query)

    for movie in movies:
        processed_movie_title = processText(movie["title"])
        if isMatch(processed_query, processed_movie_title):
            foundmovies.append(movie)
            print(f"{len(foundmovies)}. {movie['title']}")

        if len(foundmovies) == limit:
            break


def loadMovies() -> dict:
    movies_path = getPath("./data/movies.json")

    with open(movies_path) as file:
        return json.load(file)["movies"]


def processText(text: str) -> set[str]:
    lowered = text.lower()
    punctuationRemoved = removePunctuation(lowered)
    tokenizated = tokenize(punctuationRemoved)
    stopwords_removed = removeStopWords(tokenizated)

    return stopwords_removed


def isMatch(query_tokens: set[str], target_tokens: set[str]) -> bool:
    for query_token in query_tokens:
        for target_token in target_tokens:
            if query_token in target_token:
                return True

    return False


def removePunctuation(text: str) -> str:
    trans = {}
    for punctuation in string.punctuation:
        trans[punctuation] = ""

    return text.translate(str.maketrans(trans))


def tokenize(text: str) -> set[str]:
    splitted = text.split(" ")

    return set(filter(None, splitted))


def removeStopWords(words: set[str]) -> set[str]:
    stop_words = loadStopWords()

    return words.difference(stop_words)


def loadStopWords() -> set[str]:
    stop_words_path = getPath("./data/stopwords.txt")

    with open(stop_words_path, "r") as file:
        content = file.read()

    return set(content.splitlines())


def getPath(relative_path: str) -> set[str]:
    abs_path = os.path.abspath(relative_path)
    return os.path.normpath(abs_path)


if __name__ == "__main__":
    main()
