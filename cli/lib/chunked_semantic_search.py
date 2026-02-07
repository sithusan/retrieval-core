from lib.search_utils import ensure_dirs_exist, get_path, load_movies
from lib.semantic_search import SemanticSearch, semantic_chunking, cosine_similarity
from numpy.typing import NDArray
from lib.constants import SEARCH_LIMIT
import numpy as np
import json
import os


class ChunkedSemanticSearch(SemanticSearch):

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadatas = None
        self.chunk_embedding_path = get_path("cache/chunk_embeddings.npy")
        self.chunk_metadata_path = get_path("cache/chunk_metadata.json")

    def search_chunks(self, query: str, limit: int = SEARCH_LIMIT):
        query_embedding = self.generate_embedding(query)

        chunk_scores = []

        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": i,  # Need to check whether we need the chunk id of each document or global id.
                    "movie_idx": self.chunk_metadatas[i]["movie_idx"],
                    "score": score,
                }
            )

        movie_scores = {}

        for chunk_score in chunk_scores:
            if (
                not chunk_score["movie_idx"] in movie_scores
                or chunk_score["score"] > movie_scores[chunk_score["movie_idx"]]
            ):
                movie_scores[chunk_score["movie_idx"]] = chunk_score["score"]

        sorted_movie_scores = dict(
            sorted(movie_scores.items(), key=lambda item: item[1], reverse=True)
        )

        result = []

        for key, value in sorted_movie_scores.items():
            metadata = {}
            for chunk_metdata in self.chunk_metadatas:
                if chunk_metdata["movie_idx"] == key:
                    metadata = chunk_metdata
                    break

            result.append(
                {
                    "id": self.documents[key]["id"],
                    "title": self.documents[key]["title"],
                    "description": self.documents[key]["description"][:100],
                    "score": round(value, 4),
                    "metadata": metadata,
                }
            )

            if len(result) == limit:
                break

        return result

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
        self.chunk_metadatas = chunk_metadata

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
                self.chunk_metadatas = json.load(file)["chunks"]

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)


def embed_chunks() -> None:
    movies = load_movies()

    chunkedSemanticSearch = ChunkedSemanticSearch()
    chunk_embeddings = chunkedSemanticSearch.load_or_create_chunk_embeddings(movies)

    print(f"Generated {len(chunk_embeddings)} chunked embeddings")


def search_chunked(query: str, limit: int) -> None:
    try:
        movies = load_movies()

        chunkedSemanticSearch = ChunkedSemanticSearch()
        chunkedSemanticSearch.load_or_create_chunk_embeddings(movies)

        result = chunkedSemanticSearch.search_chunks(query, limit)

        for i, value in enumerate(result, 1):
            print(f"\n{i}. {value['title']} (score: {value['score']:.4f})")
            print(f"   {value['description']}...")

    except Exception as err:
        print(err)
        exit(1)
