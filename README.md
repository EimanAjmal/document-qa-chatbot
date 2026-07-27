# Document Q&A Chatbot (Work in Progress)

A RAG-based chatbot that answers questions from any uploaded PDF using local embeddings and free LLM inference.

## Tech Stack
- Python
- Sentence-Transformers (local embeddings)
- FAISS (vector search)
- Groq API (free LLM for answer generation)
- Streamlit (UI) — coming in Day 2

## Progress
- [x] PDF text extraction
- [x] Text chunking
- [x] Local embeddings generation
- [x] FAISS vector index + retrieval
- [ ] LLM-based answer generation
- [ ] Streamlit UI