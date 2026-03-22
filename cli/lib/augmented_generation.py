from lib.hybrid_search import rrf_search_with_reranking, result_from_llm
from lib.prompts import search_answer_prompt


def rag(query: str, rrf_weight: int, limit: int) -> None:
    result = rrf_search_with_reranking(query, rrf_weight, limit, None, None)
    prompt = search_answer_prompt(query, result)
    rag_answer = result_from_llm(prompt)

    print("Search Results")
    for _, v in result.items():
        print(f"-{v['title']}")
    print(rag_answer)
