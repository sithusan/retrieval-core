import os
import json
import numpy as np


def load_movies() -> list[dict]:
    movies_path = get_path("./data/movies.json")

    with open(movies_path) as file:
        return json.load(file)["movies"]


def load_stop_words() -> set[str]:
    stop_words_path = get_path("./data/stopwords.txt")

    with open(stop_words_path, "r") as file:
        return set(file.read().splitlines())


def ensure_dirs_exist(path: str = "./cache"):
    save_path = get_path(path)
    os.makedirs(save_path, exist_ok=True)


def get_path(relative_path: str) -> str:
    abs_path = os.path.abspath(relative_path)

    return os.path.normpath(abs_path)


def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)  # magnitude
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
