import os

from lib.keyword_search import InvertedIndex
from lib.chunked_semantic_search import ChunkedSemanticSearch
from lib.constants import SEARCH_LIMIT, EXTENDED_LIMIT, ALPHA, RRF_WEIGHT
from lib.search_utils import load_movies


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

    def weighted_search(
        self, query: str, alpha: float = ALPHA, limit: int = SEARCH_LIMIT
    ) -> dict:
        bm_25_search_result = self._bm25_search(query, limit * EXTENDED_LIMIT)
        chunked_semantic_search_result = self.semantic_search.search_chunks(
            query, limit * EXTENDED_LIMIT
        )

        bm_25_normalized_scores = normalize_scores(
            [v.get("score") for v in bm_25_search_result]
        )
        semantic_normalized_scores = normalize_scores(
            [v.get("score") for v in chunked_semantic_search_result]
        )

        # Hybrid search needs to include all the candidate docs.
        # Because one sided docs will still make into the result if the score is too much.
        combined_result = {}

        for i, v in enumerate(bm_25_search_result):
            document = self.idx.docmap[v["id"]]
            scores = {
                "bm25_score": v["score"],
                "normalized_bm_25_score": bm_25_normalized_scores[i],
                "semantic_score": 0.0,
                "normalized_semantic_score": 0.0,
                "hybrid_score": 0.0,
            }
            combined_result[v["id"]] = document | scores

        for i, v in enumerate(chunked_semantic_search_result):
            if v["id"] not in combined_result:
                document = self.idx.docmap[v["id"]]
                scores = {
                    "bm25_score": 0.0,
                    "normalized_bm_25_score": 0.0,
                    "semantic_score": v["score"],
                    "normalized_semantic_score": semantic_normalized_scores[i],
                    "hybrid_score": 0.0,
                }
                combined_result[v["id"]] = document | scores
            else:
                combined_result[v["id"]]["semantic_score"] = v["score"]
                combined_result[v["id"]]["normalized_semantic_score"] = (
                    semantic_normalized_scores[i]
                )

        result = {}
        for k, v in combined_result.items():
            v["hybrid_score"] = hybrid_score(
                v["normalized_bm_25_score"], v["normalized_semantic_score"], alpha
            )
            result[k] = v

        return dict(
            sorted(
                result.items(), key=lambda item: item[1]["hybrid_score"], reverse=True
            )[:limit]
        )

    def rrf_search(self, query, k, limit=10):
        bm_25_search_result = self._bm25_search(query, limit * 500)
        chunked_semantic_search_result = self.semantic_search.search_chunks(
            query, limit * 500
        )

        # Hybrid search needs to include all the candidate docs.
        # Because one sided docs will still make into the result if the score is too much.
        combined_result = {}

        for i, v in enumerate(bm_25_search_result):
            document = self.idx.docmap[v["id"]]
            scores = {
                "bm25_rank": i,
                "semantic_rank": 0,
                "rrf_score": rrf_score(i, k),
            }
            combined_result[v["id"]] = document | scores

        for i, v in enumerate(chunked_semantic_search_result):
            if v["id"] not in combined_result:
                document = self.idx.docmap[v["id"]]
                scores = {
                    "bm25_rank": 0,
                    "semantic_rank": i,
                    "rrf_score": rrf_score(i, k),
                }
                combined_result[v["id"]] = document | scores
            else:
                combined_result[v["id"]]["semantic_rank"] = i
                combined_result[v["id"]]["rrf_score"] = combined_result[v["id"]][
                    "rrf_score"
                ] + rrf_score(i, k)

        return dict(
            sorted(
                combined_result.items(),
                key=lambda item: item[1]["rrf_score"],
                reverse=True,
            )[:limit]
        )


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


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = ALPHA):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = RRF_WEIGHT):
    return 1 / (k + rank)


def weighted_search(query: str, alpha: float, limit: int) -> None:
    try:
        movies = load_movies()

        hybridSearch = HybridSearch(movies)
        result = hybridSearch.weighted_search(query, alpha, limit)

        for i, (_, v) in enumerate(result.items(), 1):
            print(f"{i}. {v['title']}")
            print(f"Hybrid Score: {v['hybrid_score']}")
            print(f"BM25: {v['bm25_score']}, Semantic :{v['semantic_score']}")
            print(v["description"][:100])

    except Exception as e:
        print(e)


def rrf_search(query: str, k: int, limit: int) -> None:
    try:
        movies = load_movies()

        hybridSearch = HybridSearch(movies)
        result = hybridSearch.rrf_search(query, k, limit)

        for i, (_, v) in enumerate(result.items(), 1):
            print(f"{i}. {v['title']}")
            print(f"RRF Score: {v['rrf_score']}")
            print(f"BM25 Rank: {v['bm25_rank']}, Semantic Rank: {v['semantic_rank']}")
            print(v["description"][:100])
    except Exception as e:
        print(e)
