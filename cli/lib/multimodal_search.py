import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from numpy.typing import NDArray
from lib.search_utils import get_path


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

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
