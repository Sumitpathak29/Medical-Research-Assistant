from pathlib import Path

import chromadb
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
# 3. Ask a question
# -------------------------

query = "What are the limitations of medical RAG systems?"


# -------------------------
# 4. Create query embedding
# -------------------------

query_embedding = embedding_model.encode(
    query
).tolist()


# -------------------------
# 5. Search ChromaDB
# -------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)


# -------------------------
# 6. Display retrieved chunks
# -------------------------

for i, document in enumerate(results["documents"][0]):
    print(f"\n--- Result {i + 1} ---")
    print(document)
