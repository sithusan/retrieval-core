import os
import pickle
import string
from nltk.stem import PorterStemmer
from lib.search_utils import (
    load_movies,
    load_stop_words,
    ensure_dirs_exist,
    get_path,
)
from collections import Counter
import math
from lib.constants import LAPLACE_SMOOTHING, BM25_K1, BM25_B


class InvertedIndex:

    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}
        self.index_path = get_path("./cache/index.pkl")
        self.docmap_path = get_path("./cache/docmap.pkl")
        self.term_frequencies_path = get_path("./cache/term_frequencies.pkl")
        self.doc_lengths_path = get_path("./cache/document_length.pkl")

    def build(self):
        movies = load_movies()

        for movie in movies:
            self.docmap[movie["id"]] = movie
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")

    def save(self):
        ensure_dirs_exist()

        with open(self.index_path, "wb") as file:
            pickle.dump(self.index, file)

        with open(self.docmap_path, "wb") as file:
            pickle.dump(self.docmap, file)

        with open(self.term_frequencies_path, "wb") as file:
            pickle.dump(self.term_frequencies, file)

        with open(self.doc_lengths_path, "wb") as file:
            pickle.dump(self.doc_lengths, file)

    def load(self):
        if (
            not os.path.isfile(self.index_path)
            or not os.path.isfile(self.docmap_path)
            or not os.path.isfile(self.term_frequencies_path)
            or not os.path.isfile(self.doc_lengths_path)
        ):
            raise RuntimeError("No file found to load")

        with open(self.index_path, "rb") as file:
            self.index = pickle.load(file)

        with open(self.docmap_path, "rb") as file:
            self.docmap = pickle.load(file)

        with open(self.term_frequencies_path, "rb") as file:
            self.term_frequencies = pickle.load(file)

        with open(self.doc_lengths_path, "rb") as file:
            self.doc_lengths = pickle.load(file)

    def get_documents(self, term: str) -> list[int]:
        token = process_term(term)
        documents = self.index.get(token, set())

        return sorted(documents)

    def get_tf(self, doc_id: int, term: str) -> int:
        token = process_term(term)
        term_frequencies = self.term_frequencies.get(doc_id, Counter())

        return term_frequencies.get(token, 0)

    def get_idf(self, term: str) -> float:
        document_ids = self.get_documents(term)

        total_document_count = len(self.docmap)
        match_document_count = len(document_ids)

        return math.log((total_document_count + 1) / (match_document_count + 1))

    def get_bm25tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)

        # May cause division by zero, if get_avg_doc_length and it's clearly say that, we haven't build our index.
        length_normalization = 1 - b + b * (doc_length / self.__get_avg_doc_length())

        return (tf * (k1 + 1)) / (tf + k1 * length_normalization)

    def get_bm25idf(self, term: str) -> float:
        document_ids = self.get_documents(term)

        total_document_count = len(self.docmap)
        match_document_count = len(document_ids)

        return math.log(
            (total_document_count - match_document_count + LAPLACE_SMOOTHING)
            / (match_document_count + LAPLACE_SMOOTHING)
            + 1
        )

    def get_bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25tf(doc_id, term) * self.get_bm25idf(term)

    def bm25_search(self, query: str, limit: int) -> list[str]:
        query_tokens = process_text(query)

        scores: dict[int, float] = {}

        for query_token in query_tokens:
            found_ids = self.get_documents(query_token)

            for found_id in found_ids:
                # aggregate the scores for the each document.
                # Case - match same document with different term
                scores[found_id] = scores.get(found_id, 0.0) + self.get_bm25(
                    found_id, query_token
                )

        sortedScores = dict(
            sorted(scores.items(), key=lambda item: item[1], reverse=True)
        )

        result = []

        for key in sortedScores:
            movie = self.docmap[key]
            score = sortedScores[key]
            movie_with_score = f"({movie['id']}) {movie['title']} - Score: {score:.2f}"

            result.append(movie_with_score)

            if len(result) == limit:
                return result

        return result

    def __add_document(self, doc_id: int, text: str):
        tokens = process_text(text)

        for token in set(tokens):
            if not token in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

        if not doc_id in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter(tokens)
        else:
            # Counter's update does not return anything and update in-place.
            self.term_frequencies[doc_id].update(tokens)

        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if len(self.docmap) == 0:
            return 0.0

        total_doc_length = 0
        total_docs = len(self.doc_lengths)

        for key in self.doc_lengths:
            total_doc_length += self.doc_lengths[key]

        return total_doc_length / total_docs


_STOP_WORDS: set[str] | None = None  # for caching


def build_command() -> None:
    print("Started building...")

    invertedIndex = InvertedIndex()
    invertedIndex.build()
    invertedIndex.save()

    print("Successfully built...")


def search_command(query: str, limit: int) -> list:
    print(f"Searching for: {query}")

    try:
        invertedIndex = InvertedIndex()
        invertedIndex.load()

        found_movies = []
        seen = set()

        query_tokens = process_text(query)

        for query_token in query_tokens:
            found_ids = invertedIndex.get_documents(query_token)
            for found_id in found_ids:
                if found_id in seen:
                    continue
                seen.add(found_id)

                found_movie = invertedIndex.docmap[found_id]
                print(
                    f"{len(found_movies) + 1 }. ID:{found_movie['id']}, Title:{found_movie['title']}"
                )
                found_movies.append(found_movie)

                if len(found_movies) == limit:
                    return found_movies

    except RuntimeError as err:
        print(err)
        exit(1)


def bm25_search_command(query: str, limit: int) -> None:
    try:
        invertedIndex = InvertedIndex()
        invertedIndex.load()

        result = invertedIndex.bm25_search(query, limit)

        for key, value in enumerate(result):
            print(f"{key+1}. {value}")

    except Exception as err:
        print(err)
        exit(1)


def tf_command(doc_id: int, term: str) -> None:
    try:
        invertedIndex = InvertedIndex()
        invertedIndex.load()
        tf = invertedIndex.get_tf(doc_id, term)

        print(f"Term {term}'s frequency is {tf}")
    except Exception as err:
        print(err)
        exit(1)


def idf_command(term: str) -> None:
    try:
        invertedIndex = InvertedIndex()
        invertedIndex.load()
        idf = invertedIndex.get_idf(term)

        print(f"Inverse document frequency of '{term}': {idf:.2f}")
    except Exception as e:
        print(e)


def tf_idf_command(doc_id: int, term: str) -> None:
    try:

        invertedIndex = InvertedIndex()
        invertedIndex.load()
        tf = invertedIndex.get_tf(doc_id, term)
        idf = invertedIndex.get_idf(term)

        tf_idf = tf * idf

        print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")
    except Exception as e:
        print(e)


def bm25idf_command(term: str) -> None:
    try:
        invertedIndex = InvertedIndex()
        invertedIndex.load()
        bm25idf = invertedIndex.get_bm25idf(term)

        print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")
    except Exception as e:
        print(e)


# prevent the term saturation
def bm25tf_command(doc_id: int, term: str, k1: float, b: float) -> None:
    try:
        invertedIndex = InvertedIndex()
        invertedIndex.load()
        bm25tf = invertedIndex.get_bm25tf(doc_id, term, k1, b)

        print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")
    except Exception as e:
        print(e)


def process_text(text: str) -> list[str]:
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


def tokenize(text: str) -> list[str]:
    splitted = text.split()

    return list(filter(None, splitted))


def remove_stopwords(words: list[str]) -> list[str]:
    global _STOP_WORDS

    if _STOP_WORDS is None:
        _STOP_WORDS = load_stop_words()

    filtered_words = []

    for word in words:
        if word not in _STOP_WORDS:
            filtered_words.append(word)

    return filtered_words


def stem(words: list[str]) -> list[str]:
    stemmer = PorterStemmer()

    stemmed = []

    for word in words:
        stemmed.append(stemmer.stem(word=word))

    return stemmed


def is_match(query_tokens: set[str], target_tokens: set[str]) -> bool:
    for query_token in query_tokens:
        for target_token in target_tokens:
            if query_token in target_token:
                return True
    return False


def process_term(term: str) -> str:
    tokens = process_text(term)

    if len(tokens) != 1:
        raise ValueError("Term MUST be exactly one")

    return tokens[0]
