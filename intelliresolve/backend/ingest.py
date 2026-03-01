import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load complaints data
with open("../data/complaints.json", "r") as f:
    complaints = json.load(f)

# Generate embeddings
embeddings = model.encode(complaints)

# Save embeddings locally (temporary storage before Endee integration)
np.save("complaint_embeddings.npy", embeddings)

print("✅ Embeddings generated and saved successfully!")