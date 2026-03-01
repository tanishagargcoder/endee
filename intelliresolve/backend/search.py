import json
from sentence_transformers import SentenceTransformer
from endee_client import EndeeClient

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load complaints
with open("../data/complaints.json", "r") as f:
    complaints = json.load(f)

# Generate embeddings again (simulate DB load)
embeddings = model.encode(complaints)

# Initialize Endee client
endee = EndeeClient()
endee.insert(embeddings, complaints)

def semantic_search(query, top_k=3):
    query_vector = model.encode([query])
    results = endee.search(query_vector, top_k)

    print("\n🔍 Most Similar Complaints:\n")
    for r in results:
        print(f"Complaint: {r['complaint']}")
        print(f"Similarity Score: {r['score']:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    user_query = input("Enter complaint search query: ")
    semantic_search(user_query)