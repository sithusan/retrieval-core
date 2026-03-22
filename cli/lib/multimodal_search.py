import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from numpy.typing import NDArray
from lib.search_utils import get_path, load_movies, cosine_similarity


class MultimodalSearch:
    def __init__(self, docs: list[dict] = [{}], model_name="clip-ViT-B-32"):
        self.docs = docs
        self.model = SentenceTransformer(model_name)
        self.texts = []
        for doc in self.docs:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def search_with_image(self, url: str):
        image_embeddings = self.embed_image(url)
        similarities = []

        for i, embedding in enumerate(self.text_embeddings):
            document = self.docs[i]
            document["score"] = cosine_similarity(embedding, image_embeddings)
            similarities.append(document)

        return sorted(similarities, key=lambda item: item["score"], reverse=True)

    def embed_image(self, url: str) -> NDArray[np.floating]:
        url = get_path(url)
        image = Image.open(url)

        return self.model.encode([image])[0]


def verify_image_embedding(url: str) -> str:
    try:
        multimodal_search = MultimodalSearch()
        embedding = multimodal_search.embed_image(url)

        print(f"Embedding shape: {embedding.shape[0]} dimensions")
    except Exception as err:
        print(err)
        exit(1)


def image_search(url: str) -> None:
    try:
        display_limit = 5
        movies = load_movies()
        multimodal_search = MultimodalSearch(movies)
        found_movies = multimodal_search.search_with_image(url)

        for i, found_movie in enumerate(found_movies, 1):
            if i > display_limit:
                break

            print(
                f"{i}.{found_movie['title']} (similarity: {found_movie['score']:.3f})"
            )
            print(f"{found_movie["description"][:100]}...")

    except Exception as err:
        print(err)
        exit(1)
