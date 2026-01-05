import json
import os
import pickle


class InvertedIndex:

    def __init__(self):
        self.index: dict[str, list[int]] = {}
        self.docmap: dict[int, dict] = {}

    def build(self):
        movies = self.__load_movies()

        for movie in movies:
            self.docmap[movie["id"]] = movie
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")

    def save(self):
        savePath = self.__get_path("./cache")
        os.makedirs(savePath, exist_ok=True)

        indexPath = self.__get_path("./cache/index.pkl")
        docmapPath = self.__get_path("./cache/docmap.pkl")

        with open(indexPath, "wb") as file:
            pickle.dump(self.index, file)

        with open(docmapPath, "wb") as file:
            pickle.dump(self.docmap, file)

    def get_documents(self, term: str) -> list:
        documents = self.index.get(term.lower(), [])

        return sorted(documents)

    def __add_document(self, doc_id: int, text: str):
        tokens = self.__tokenize(text)

        for token in tokens:
            if not token in self.index:
                self.index[token] = []
            self.index[token].append(doc_id)

    def __tokenize(self, text: str) -> list[str]:
        splitted = text.lower().split(" ")

        return list(filter(None, splitted))

    def __load_movies(self) -> list[dict]:
        movies_path = self.__get_path("./data/movies.json")

        with open(movies_path) as file:
            return json.load(file)["movies"]

    def __get_path(self, relative_path: str) -> str:
        abs_path = os.path.abspath(relative_path)
        return os.path.normpath(abs_path)
