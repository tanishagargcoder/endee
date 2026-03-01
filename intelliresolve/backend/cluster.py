import json
import numpy as np
from sklearn.cluster import KMeans

# Load complaints
with open("../data/complaints.json", "r") as f:
    complaints = json.load(f)

# Load embeddings
embeddings = np.load("complaint_embeddings.npy")

# Apply KMeans clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(embeddings)

labels = kmeans.labels_

print("\n📊 Complaint Clusters:\n")

for cluster_id in range(num_clusters):
    print(f"\n🔹 Cluster {cluster_id + 1}")
    print("-" * 40)
    for idx, label in enumerate(labels):
        if label == cluster_id:
            print(complaints[idx])