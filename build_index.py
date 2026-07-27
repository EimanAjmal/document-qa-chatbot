from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# ---- STEP A: Extract text from PDF ----
print("Reading PDF...")
reader = PdfReader("test.pdf")

full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

print(f"Total characters extracted: {len(full_text)}")

# ---- STEP B: Divide text into chunks----
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(full_text)
print(f"Total chunks created: {len(chunks)}")

# ---- STEP C: Convert every chunk into embedding ----
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Creating embeddings for all chunks...")
embeddings = model.encode(chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

# ---- STEP D: FAISS index made and save ----
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "faiss_index.bin")

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Done! Index and chunks saved successfully.")