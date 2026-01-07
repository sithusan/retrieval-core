import os
import json

DEFAULT_SEARCH_LIMIT = 5


def load_movies() -> list[dict]:
    movies_path = get_path("./data/movies.json")

    with open(movies_path) as file:
        return json.load(file)["movies"]


def load_stop_words() -> set[str]:
    stop_words_path = get_path("./data/stopwords.txt")

    with open(stop_words_path, "r") as file:
        return set(file.read().splitlines())


def get_path(relative_path: str) -> str:
    abs_path = os.path.abspath(relative_path)

    return os.path.normpath(abs_path)
