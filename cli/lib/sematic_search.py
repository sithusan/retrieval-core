from sentence_transformers import SentenceTransformer


class SematicSearch:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")


def verifyModel() -> None:
    sematicSearch = SematicSearch()
    print(f"Model loaded: {sematicSearch.model}")
    print(f"Max sequence length: {sematicSearch.model.max_seq_length}")
