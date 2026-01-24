from sentence_transformers import SentenceTransformer


class SematicSearch:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embedding(self, text: str) -> list:
        if len(text.strip()) == 0:
            raise ValueError("The provided text is empty")

        return self.model.encode([text])[0]


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
    
