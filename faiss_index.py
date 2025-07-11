import faiss
import numpy as np
import pickle

class FaissIndex:
    def __init__(self, dim=768):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine sim with normalized vectors
        self.id_map = []  # stores metadata like (doc_id, chunk_text)

    def add_embeddings(self, embeddings: np.ndarray, metadata: list):
        """
        embeddings: np.ndarray shape=(n, dim), must be normalized
        metadata: List of tuples (doc_id, chunk_text)
        """
        assert embeddings.shape[0] == len(metadata), "Embeddings and metadata length mismatch"
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.id_map.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k=5):
        faiss.normalize_L2(query_embedding)
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for i in I[0]:
            if i == -1:
                continue
            results.append(self.id_map[i])
        return results

    def save(self, path):
        faiss.write_index(self.index, path + ".index")
        with open(path + ".pkl", "wb") as f:
            pickle.dump(self.id_map, f)

    @classmethod
    def load(cls, path):
        instance = cls()
        instance.index = faiss.read_index(path + ".index")
        with open(path + ".pkl", "rb") as f:
            instance.id_map = pickle.load(f)
        instance.dim = instance.index.d
        return instance
