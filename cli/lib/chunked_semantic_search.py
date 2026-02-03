from lib.search_utils import ensure_dirs_exist, get_path, load_movies
from lib.semantic_search import SemanticSearch, semantic_chunking
from numpy.typing import NDArray
from lib.constants import SEARCH_LIMIT
import numpy as np
import json
import os


class ChunkedSematicSearch(SemanticSearch):

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embedding_path = get_path("cache/chunk_embeddings.npy")
        self.chunk_metadata_path = get_path("cache/chunk_metadata.json")

    def search_chunks(self, query: str, limit: int = SEARCH_LIMIT):
        print("search chunks")

    def build_chunk_embeddings(self, documents: list[dict]) -> NDArray[np.floating]:
        self.documents = documents

        all_chunks = []
        chunk_metadata = []

        for i, document in enumerate(documents):
            self.document_map[document["id"]] = document

            if len(document["description"]) == 0:
                continue

            chunks = semantic_chunking(document["description"], 4, 1)
            total_chunks = len(chunks)

            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {"movie_idx": i, "chunk_idx": j, "total_chunks": total_chunks}
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        ensure_dirs_exist()

        with open(self.chunk_embedding_path, "wb") as file:
            np.save(file, self.chunk_embeddings)

        with open(self.chunk_metadata_path, "w") as file:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)},
                file,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(
        self, documents: list[dict]
    ) -> NDArray[np.floating]:
        self.documents = documents

        for document in documents:
            self.document_map[document["id"]] = document

        if os.path.exists(self.chunk_embedding_path) and os.path.exists(
            self.chunk_metadata_path
        ):
            with open(self.chunk_embedding_path, "rb") as file:
                self.chunk_embeddings = np.load(file)

            with open(self.chunk_metadata_path, "r") as file:
                self.chunk_metadata = json.load(file)["chunks"]

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)


def embed_chunks() -> None:
    movies = load_movies()

    chunkedSematicSearch = ChunkedSematicSearch()
    chunk_embeddings = chunkedSematicSearch.load_or_create_chunk_embeddings(movies)

    print(f"Generated {len(chunk_embeddings)} chunked embeddings")


def search_chunked(query: str, limit: int) -> None:
    try:
        movies = load_movies()

        chunkedSematicSearch = ChunkedSematicSearch()
        chunkedSematicSearch.load_or_create_embeddings(movies)
        chunkedSematicSearch.search_chunks(query, limit)

    except Exception as err:
        print(err)
        exit(1)
