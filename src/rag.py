from pathlib import Path

import chromadb
import ollama
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_DIR / "chroma_db"


# -------------------------
# 1. Load embedding model
# -------------------------

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------
# 2. Connect to ChromaDB
# -------------------------

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_collection(
    name="medical_documents"
)


# -------------------------
# 3. User question
# -------------------------

query = "What are the limitations of medical RAG systems?"


# -------------------------
# 4. Embed the question
# -------------------------

query_embedding = embedding_model.encode(
    query
).tolist()


# -------------------------
# 5. Retrieve relevant chunks
# -------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

retrieved_chunks = results["documents"][0]


# -------------------------
# 6. Combine retrieved context
# -------------------------

context = "\n\n".join(retrieved_chunks)


# -------------------------
# 7. Build prompt
# -------------------------

prompt = f"""
You are a medical research assistant.

Answer the user's question using the provided context.

If the context does not contain enough information to answer the question,
say that you do not have enough information from the provided context.

User question:
{query}

Relevant context:
{context}

Answer:
"""


# -------------------------
# 8. Ask Llama
# -------------------------

response = ollama.generate(
    model="llama3.2:1b",
    prompt=prompt
)


# -------------------------
# 9. Print answer
# -------------------------

print("\n--- Answer ---")
print(response["response"])
