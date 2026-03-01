import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load complaints
with open("../data/complaints.json", "r") as f:
    complaints = json.load(f)

# Load stored embeddings
embeddings = np.load("complaint_embeddings.npy")

def semantic_search(query, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    print("\n🔍 Most Similar Complaints:\n")
    for idx in top_indices:
        print(f"Complaint: {complaints[idx]}")
        print(f"Similarity Score: {similarities[idx]:.4f}")
        print("-" * 40)

# Test query
if __name__ == "__main__":
    user_query = input("Enter complaint search query: ")
    semantic_search(user_query)