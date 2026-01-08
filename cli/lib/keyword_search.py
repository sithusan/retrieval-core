import os
import pickle
import string
from nltk.stem import PorterStemmer
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stop_words,
    get_path,
)


class InvertedIndex:

    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.index_path = get_path("./cache/index.pkl")
        self.docmap_path = get_path("./cache/docmap.pkl")

    def build(self):
        movies = load_movies()

        for movie in movies:
            self.docmap[movie["id"]] = movie
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")

    def save(self):
        save_path = get_path("./cache")
        os.makedirs(save_path, exist_ok=True)

        with open(self.index_path, "wb") as file:
            pickle.dump(self.index, file)

        with open(self.docmap_path, "wb") as file:
            pickle.dump(self.docmap, file)

    def get_documents(self, term: str) -> list:
        documents = self.index.get(term.lower(), set())

        return sorted(documents)

    def load(self):
        if not os.path.isfile(self.index_path) or not os.path.isfile(self.docmap_path):
            raise RuntimeError("No file found to load")

        with open(self.index_path, "rb") as file:
            self.index = pickle.load(file)

        with open(self.docmap_path, "rb") as file:
            self.docmap = pickle.load(file)

    def __add_document(self, doc_id: int, text: str):
        tokens = process_text(text)

        for token in tokens:
            if not token in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)


_STOP_WORDS: set[str] | None = None  # for caching


def build_command() -> None:
    invertedIndex = InvertedIndex()
    invertedIndex.build()
    invertedIndex.save()

    invertedIndex.get_documents("merida")


def search_command(query: str) -> list:
    print(f"Searching for: {query}")

    try:
        invertedIndex = InvertedIndex()
        invertedIndex.load()

        found_movies = []
        query_tokens = process_text(query)

        for query_token in query_tokens:
            found_ids = invertedIndex.get_documents(query_token)
            for found_id in found_ids:
                found_movie = invertedIndex.docmap[found_id]
                print(
                    f"{len(found_movies) + 1 }. ID:{found_movie['id']}, Title:{found_movie['title']}"
                )
                found_movies.append(found_movie)

                if len(found_movies) == DEFAULT_SEARCH_LIMIT:
                    return found_movies

    except RuntimeError as err:
        print(err)
        exit(1)


def process_text(text: str) -> set[str]:
    lowered = text.lower()
    punctuation_removed = remove_punctuation(lowered)
    tokenizated = tokenize(punctuation_removed)
    stopwords_removed = remove_stopwords(tokenizated)
    stemmed = stem(stopwords_removed)

    return stemmed


def remove_punctuation(text: str) -> str:
    trans = {}
    for punctuation in string.punctuation:
        trans[punctuation] = ""

    return text.translate(str.maketrans(trans))


def tokenize(text: str) -> set[str]:
    splitted = text.split(" ")

    return set(filter(None, splitted))


def remove_stopwords(words: set[str]) -> set[str]:
    global _STOP_WORDS

    if _STOP_WORDS is None:
        _STOP_WORDS = load_stop_words()

    return words.difference(_STOP_WORDS)


def stem(words: set[str]) -> set[str]:
    stemmer = PorterStemmer()

    stemmed = set()

    for word in words:
        stemmed.add(stemmer.stem(word=word))

    return stemmed


def is_match(query_tokens: set[str], target_tokens: set[str]) -> bool:
    for query_token in query_tokens:
        for target_token in target_tokens:
            if query_token in target_token:
                return True
    return False
