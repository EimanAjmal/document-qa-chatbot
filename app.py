import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import faiss
import numpy as np
import pickle
import os
from pypdf import PdfReader

load_dotenv()

st.set_page_config(page_title="Document Q&A Chatbot", page_icon="📄")
st.title("📄 Document Q&A Chatbot")
st.write("Upload a PDF and ask questions about it.")

# ---- Load models (cached so it doesn't reload every time) ----
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedding_model()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---- Chunking function (same as before) ----
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

# ---- File upload ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Only rebuild the index if this is a new file
    if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
        with st.spinner("Reading and processing PDF..."):
            reader = PdfReader(uploaded_file)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"

            chunks = chunk_text(full_text)
            embeddings = embed_model.encode(chunks)
            embeddings = np.array(embeddings).astype('float32')

            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)

            # Save in session so we don't reprocess on every question
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.processed_file = uploaded_file.name

        st.success(f"Processed {len(st.session_state.chunks)} chunks from the document!")

   # ---- Display existing chat history ----
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            with st.expander("See source chunks used"):
                for i, chunk in enumerate(entry["chunks"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(chunk[:300] + "...")

    # ---- New question input (chat-style) ----
    question = st.chat_input("Ask a question about the document:")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                question_embedding = embed_model.encode([question]).astype('float32')
                distances, indices = st.session_state.index.search(question_embedding, k=3)
                relevant_chunks = [st.session_state.chunks[i] for i in indices[0]]
                context = "\n\n".join(relevant_chunks)

                prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

                response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.choices[0].message.content

            st.write(answer)
            with st.expander("See source chunks used"):
                for i, chunk in enumerate(relevant_chunks):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(chunk[:300] + "...")

        # Save this Q&A to history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "chunks": relevant_chunks
        })