from sentence_transformers import SentenceTransformer
from lib.search_utils import get_path, ensure_dirs_exist, load_movies
from numpy.typing import NDArray
import numpy as np
import os
from lib.constants import SEARCH_LIMIT


class SemanticSearch:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings: NDArray[np.floating] = None
        self.documents: list[dict] = None
        self.document_map: dict[int, dict] = {}
        self.movie_embedding_path = get_path("./cache/movie_embeddings.npy")

    def search(self, query: str, limit: int = SEARCH_LIMIT):
        if len(self.embeddings) == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarities = []

        for i, embedding in enumerate(
            self.embeddings
        ):  # model's encode guaranteed that, input list order is preserved, output embeddings are in the same order.
            document = self.documents[i]
            score = cosine_similarity(query_embedding, embedding)

            similarities.append((score, document))

        sorted_similarities = sorted(
            similarities, key=lambda item: item[0], reverse=True
        )

        return sorted_similarities[:limit]

    def generate_embedding(self, text: str) -> NDArray[np.floating]:
        if len(text.strip()) == 0:
            raise ValueError("The provided text is empty")

        return self.model.encode([text])[0]

    def load_or_create_embeddings(self, documents: list[dict]) -> NDArray[np.floating]:
        self.documents = documents

        for document in documents:
            self.document_map[document["id"]] = document

        if os.path.exists(self.movie_embedding_path):
            with open(self.movie_embedding_path, "rb") as file:
                self.embeddings = np.load(file)

            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def build_embeddings(self, documents: list[dict]) -> NDArray[np.floating]:
        self.documents = documents

        movies = []
        for document in documents:
            self.document_map[document["id"]] = document
            movies.append(f"{document['title']}: {document['description']}")

        self.embeddings = self.model.encode(movies, show_progress_bar=True)

        ensure_dirs_exist()

        with open(self.movie_embedding_path, "wb") as file:
            np.save(file, self.embeddings)

        return self.embeddings


def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)  # magnitude
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def verify_model() -> None:
    semanticSearch = SemanticSearch()
    print(f"Model loaded: {semanticSearch.model}")
    print(f"Max sequence length: {semanticSearch.model.max_seq_length}")


def embed_text(text: str) -> None:
    semanticSearch = SemanticSearch()
    embedding = semanticSearch.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str) -> None:
    semanticSearch = SemanticSearch()
    embedding = semanticSearch.generate_embedding(query)

    print(f"Text: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    movies = load_movies()

    semanticSearch = SemanticSearch()
    embeddings = semanticSearch.load_or_create_embeddings(movies)

    print(f"Number of docs:   {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def search(query: str, limit: int) -> None:
    try:
        movies = load_movies()

        semanticSearch = SemanticSearch()
        semanticSearch.load_or_create_embeddings(movies)
        result = semanticSearch.search(query, limit)

        for i, [score, document] in enumerate(result):
            print(f"{i+1}. {document['title']} (score: {score})")
            print(f"{document["description"]} \n")
    except Exception as err:
        print(err)
        exit(1)


def chunk(text: str, chunk_size: int) -> None:
    words = text.split(" ")

    result = []
    remaining = len(words)
    current = 0

    while remaining > chunk_size:
        result.append(" ".join(words[current : current + chunk_size]))
        current += chunk_size
        remaining -= chunk_size

    result.append(" ".join(words[current:]))

    print(f"Chunking {len(text)} characters")
    for i, value in enumerate(result, 1):
        print(f"{i}. {value}")
