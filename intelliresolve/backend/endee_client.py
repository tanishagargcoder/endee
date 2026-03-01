import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EndeeClient:
    """
    Simulated Endee Vector Database Client
    In production, this would connect to Endee via API.
    """

    def __init__(self):
        self.vectors = None
        self.metadata = None

    def insert(self, embeddings, metadata):
        """
        Insert vectors into vector database
        """
        self.vectors = embeddings
        self.metadata = metadata
        print("✅ Vectors inserted into Endee (simulated)")

    def search(self, query_vector, top_k=3):
        """
        Perform vector similarity search
        """
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "complaint": self.metadata[idx],
                "score": float(similarities[idx])
            })

        return results