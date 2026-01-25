from sentence_transformers import SentenceTransformer
from lib.search_utils import get_path, ensure_dirs_exist, load_movies
import numpy as np
import os


class SematicSearch:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # TODO:: add type
        self.embeddings = None
        self.documents: list[dict] = None
        self.documentMap: dict[int, dict] = {}
        self.movie_embedding_path = get_path("./cache/movie_embeddings.npy")

    def generate_embedding(self, text: str) -> list:
        if len(text.strip()) == 0:
            raise ValueError("The provided text is empty")

        return self.model.encode([text])[0]

    # TODO:: add return type
    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents

        for document in documents:
            self.documentMap[document["id"]] = document

        if os.path.exists(self.movie_embedding_path):
            with open(self.movie_embedding_path, "rb") as file:
                self.embeddings = np.load(file)

            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    # TODO:: add return type
    def build_embeddings(self, documents: list[dict]):
        self.documents = documents

        movies = []
        for document in documents:
            self.documentMap[document["id"]] = document
            movies.append(f"{document['title']}: {document['description']}")

        self.embeddings = self.model.encode(movies, show_progress_bar=True)

        ensure_dirs_exist()

        with open(self.movie_embedding_path, "wb") as file:
            np.save(file, self.embeddings)

        return self.embeddings


def verify_model() -> None:
    sematicSearch = SematicSearch()
    print(f"Model loaded: {sematicSearch.model}")
    print(f"Max sequence length: {sematicSearch.model.max_seq_length}")


def embed_text(text: str):
    sematicSearch = SematicSearch()
    embedding = sematicSearch.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    movies = load_movies()

    sematicSearch = SematicSearch()
    embeddings = sematicSearch.load_or_create_embeddings(movies)

    print(f"Number of docs:   {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )
