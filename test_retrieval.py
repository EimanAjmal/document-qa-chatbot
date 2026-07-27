from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Saved index and chunks load 
index = faiss.read_index("faiss_index.bin")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Write your question here
question = "What is the main topic discussed in the document?"

# Convert Question into embedding
question_embedding = model.encode([question]).astype('float32')

# Find FAISS most relevant top 3 chunks 
distances, indices = index.search(question_embedding, k=3)

print(f"Question: {question}\n")
print("Top matching chunks:\n")
for i, idx in enumerate(indices[0]):
    print(f"--- Match {i+1} (distance: {distances[0][i]:.2f}) ---")
    print(chunks[idx][:300])  # sirf pehle 300 characters dikhao
    print()