from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Test 1: Free local embeddings
print("Loading embedding model (first time takes a minute to download)...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
test_embedding = embed_model.encode("Hello, this is a test.")
print("Embedding success! Length:", len(test_embedding))

# Test 2: Free Groq chat
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)
print("Groq response:", response.choices[0].message.content)