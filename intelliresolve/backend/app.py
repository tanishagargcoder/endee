import streamlit as st
import json
import os
import pandas as pd
import time
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from endee_client import EndeeClient

# ==========================
# PAGE CONFIG + STYLING
# ==========================
st.set_page_config(page_title="IntelliResolve", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0f172a;
}
h1, h2, h3 {
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 IntelliResolve")
st.subheader("AI-Powered Complaint Clustering & Semantic Search")
st.success("🚀 System powered by Endee Vector Architecture")

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ==========================
# LOAD DATA
# ==========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "complaints.json")

with open(data_path, "r") as f:
    complaints = json.load(f)

# 🔥 Scale simulation (realistic demo size)
complaints = complaints * 50  # simulate larger dataset

embeddings = model.encode(complaints)

# ==========================
# ENDEE CLIENT
# ==========================
endee = EndeeClient()
endee.insert(embeddings, complaints)

# ==========================
# METRICS
# ==========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Complaints", len(complaints))
col2.metric("Embedding Dimensions", len(embeddings[0]))
col3.metric("Clusters Configured", 3)
col4.metric("Vector DB Type", "Semantic")

# ==========================
# SAMPLE DATA VIEWER
# ==========================
st.markdown("## 📋 Sample Complaint Data")

if st.button("Show Random Sample"):
    st.write(pd.DataFrame(complaints[:10], columns=["Complaint"]))

# ==========================
# MODE TOGGLE
# ==========================
mode = st.radio("Select Mode:", ["Semantic Search", "Cluster Overview"])

# ==========================
# SEMANTIC SEARCH MODE
# ==========================
if mode == "Semantic Search":

    st.markdown("## 🔍 Semantic Complaint Search")

    query = st.text_input("Enter complaint search query:")

    if query:
        query_vector = model.encode([query])

        start_time = time.time()
        results = endee.search(query_vector, top_k=3)
        end_time = time.time()

        st.caption(f"⏱ Search completed in {end_time - start_time:.4f} seconds")

        st.markdown("### 📊 Most Similar Complaints")

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        # Similarity progress bars
        for r in results:
            st.write(f"**{r['complaint']}**")
            st.progress(float(r['score']))
            st.write(f"Similarity Score: {r['score']:.4f}")
            st.write("---")

        # CSV export
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Search Results as CSV",
            csv,
            "search_results.csv",
            "text/csv"
        )

    # Live Complaint Analysis
    st.markdown("## ➕ Analyze New Complaint")

    new_complaint = st.text_input("Enter a new complaint to analyze:")

    if new_complaint:
        new_embedding = model.encode([new_complaint])
        closest = endee.search(new_embedding, top_k=1)[0]

        st.write("### 🔎 Closest Existing Issue:")
        st.write(closest["complaint"])
        st.progress(float(closest["score"]))
        st.write(f"Similarity Score: {closest['score']:.4f}")

# ==========================
# CLUSTER MODE
# ==========================
elif mode == "Cluster Overview":

    st.markdown("## 📂 Complaint Clusters")

    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)

    labels = kmeans.labels_
    cluster_data = {}

    for idx, label in enumerate(labels):
        cluster_data.setdefault(label, []).append(complaints[idx])

    # Auto cluster labeling
    def auto_label(cluster_items):
        text = " ".join(cluster_items).lower()
        if "payment" in text or "checkout" in text:
            return "💳 Payment Issues"
        elif "login" in text or "password" in text:
            return "🔐 Login Issues"
        elif "delivery" in text or "order" in text:
            return "📦 Delivery Issues"
        else:
            return "⚠️ Other Issues"

    cluster_counts = {}

    for cluster_id, items in cluster_data.items():
        label_name = auto_label(items)
        st.markdown(f"### {label_name}")
        for item in items[:10]:  # limit display for readability
            st.write("-", item)
        cluster_counts[label_name] = len(items)

    # Issue Distribution Chart
    st.markdown("## 📈 Issue Distribution")

    df_counts = pd.DataFrame(
        list(cluster_counts.items()),
        columns=["Issue Category", "Number of Complaints"]
    )

    st.bar_chart(df_counts.set_index("Issue Category"))

# ==========================
# ARCHITECTURE SECTION
# ==========================
st.markdown("## 🏗 System Architecture")

st.code("""
User Query
   ↓
SentenceTransformer Embedding
   ↓
Endee Vector Layer
   ↓
Similarity Search
   ↓
Cluster Analysis
   ↓
Streamlit Dashboard
""")