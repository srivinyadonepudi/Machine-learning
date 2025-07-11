import numpy as np

class Retriever:
    def __init__(self, faiss_index):
        self.index = faiss_index

    def retrieve(self, query_embedding: np.ndarray, top_k=5):
        """
        Returns List of (doc_id, chunk_text)
        """
        query_embedding = np.array([query_embedding]).astype('float32')
        results = self.index.search(query_embedding, top_k)
        return results
