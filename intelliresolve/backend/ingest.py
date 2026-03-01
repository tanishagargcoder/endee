import json
from sentence_transformers import SentenceTransformer
from endee_client import EndeeClient

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load complaints
with open("../data/complaints.json", "r") as f:
    complaints = json.load(f)

# Generate embeddings
embeddings = model.encode(complaints)

# Insert into Endee
endee = EndeeClient()
endee.insert(embeddings, complaints)

print("🚀 Complaints successfully indexed into Endee")