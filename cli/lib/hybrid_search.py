import os
import time

from lib.keyword_search import InvertedIndex
from lib.chunked_semantic_search import ChunkedSemanticSearch
from lib.constants import SEARCH_LIMIT, EXTENDED_LIMIT, ALPHA, RRF_WEIGHT
from lib.search_utils import load_movies
from lib.prompts import (
    get_spell_correcter_prompt,
    get_query_rewriter_prompt,
    get_query_expander_prompt,
    get_rerank_prompt,
    get_batch_rerank_prompt,
    get_evaluation_prompt,
)
from dotenv import load_dotenv
from google import genai
import json
from sentence_transformers import CrossEncoder


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
                "semantic_rank": None,
                "rrf_score": rrf_score(i, k),
            }
            combined_result[v["id"]] = document | scores

        for i, v in enumerate(chunked_semantic_search_result):
            if v["id"] not in combined_result:
                document = self.idx.docmap[v["id"]]
                scores = {
                    "bm25_rank": None,
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


def rrf_search(
    query: str,
    k: int,
    limit: int,
    enhance: str | None,
    rerank_method: str | None,
    evaluate: bool,
) -> None:
    try:
        result = rrf_search_with_reranking(query, k, limit, enhance, rerank_method)
        formatted_rrf_search_result = format_rrf_search_result(result)

        for v in formatted_rrf_search_result:
            print(v)

        if evaluate:
            prompt = get_evaluation_prompt(query, formatted_rrf_search_result)
            evaluations_json = result_from_llm(prompt)
            evaluations = json.loads(evaluations_json)

            for i, (_, v) in enumerate(result.items()):
                print(f"{i+1}. {v['title']}: {evaluations[i]}/3")
    except Exception as e:
        print(e)


def format_rrf_search_result(rrf_search_result: dict) -> list[str]:
    formatted_result = []

    for i, (_, v) in enumerate(rrf_search_result.items(), 1):
        formatted_result.append(
            f"{i}. {v['title']}\nRRF Score: {v['rrf_score']}\nBM25 Rank: {v['bm25_rank']}, Semantic Rank: {v['semantic_rank']}\n{v['description'][:100]}"
        )
    return formatted_result


def rrf_search_with_reranking(
    query: str, k: int, limit: int, enhance: str | None, rerank_method: str | None
) -> dict:
    enhanced_query = get_enhanced_query(query, enhance)
    rerank_limit = get_rerank_limit(limit, rerank_method)
    movies = load_movies()

    hybridSearch = HybridSearch(movies)
    rrf_result = hybridSearch.rrf_search(enhanced_query, k, rerank_limit)

    match rerank_method:
        case "individual":
            return result_from_individual_rerank(rrf_result, query, limit)
        case "batch":
            return result_from_batch_rerank(rrf_result, query, limit)
        case "cross_encoder":
            return result_from_cross_encoder_rerank(rrf_result, query, limit)

    return rrf_result


def get_enhanced_query(query: str, enhance: str | None) -> str:
    if not enhance:
        return query

    match enhance:
        case "spelling":
            prompt = get_spell_correcter_prompt(query)
        case "rewrite":
            prompt = get_query_rewriter_prompt(query)
        case "expand":
            prompt = get_query_expander_prompt(query)

    enhanced_query = result_from_llm(prompt)
    print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")

    return enhanced_query


def result_from_individual_rerank(
    initial_search_result: dict, query: str, limit: int
) -> dict:
    result = {}
    for k, v in initial_search_result.items():
        time.sleep(3)
        prompt = get_rerank_prompt(query, v)
        v["rerank_score"] = result_from_llm(prompt)
        result[k] = v

    return dict(
        list(
            sorted(
                result.items(),
                key=lambda item: item[1]["rerank_score"],
                reverse=True,
            )[:limit]
        )
    )


def result_from_batch_rerank(
    initial_search_result: dict, query: str, limit: int
) -> dict:
    doc_list_str = ",".join(str(value) for value in initial_search_result.values())
    prompt = get_batch_rerank_prompt(query, doc_list_str)

    movie_ids_json = result_from_llm(prompt)
    movie_ids = json.loads(movie_ids_json)

    result = {}

    for i, movie_id in enumerate(movie_ids, 0):
        if i == limit:
            return result
        result[movie_id] = initial_search_result.get(movie_id)

    return result


def result_from_cross_encoder_rerank(
    initial_search_result: dict, query: str, limit: int
) -> dict:
    pairs = []

    for _, v in initial_search_result.items():
        pairs.append([query, f"{v['title']} - {v['description']}"])

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)

    search_result_with_score = {}

    for i, (k, v) in enumerate(initial_search_result.items()):
        v["cross_encoder_score"] = scores[i]
        search_result_with_score[k] = v

    return dict(
        list(
            sorted(
                search_result_with_score.items(),
                key=lambda item: item[1]["cross_encoder_score"],
                reverse=True,
            )[:limit]
        )
    )


def get_rerank_limit(limit: int, rerank_method: str | None):
    if not rerank_method:
        return limit

    match rerank_method:
        case "individual":
            return limit * 5
        case "batch":
            return limit * 5
        case "cross_encoder":
            return limit * 5


def result_from_llm(prompt: str) -> str:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    result = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return result.text
