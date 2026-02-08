import os

from lib.keyword_search import InvertedIndex
from lib.chunked_semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize(scores: list[float]) -> None:
    normalized_scores = normalize_scores(scores)

    for normalized_score in normalized_scores:
        print(f"* {normalized_score:.4f}")


def normalize_scores(scores: list[float]) -> list[float]:
    if len(scores) == 0:
        return []

    min_score = scores[0]
    max_score = scores[0]

    for score in scores:
        if score < min_score:
            min_score = score
        if score > max_score:
            max_score = score

    if min_score == max_score:
        return [1.0] * len(scores)

    result = []
    for score in scores:
        normalized_score = (score - min_score) / (max_score - min_score)
        result.append(normalized_score)

    return result


def weighted_search() -> None:
    print("This is weighted search")
