from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# -------------------------
# 1. Paths
# -------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_DIR / "chroma_db"


# -------------------------
# 2. Load embedding model
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------
# 3. Load ChromaDB
# -------------------------

vectorstore = Chroma(
    persist_directory=str(CHROMA_PATH),
    collection_name="medical_documents",
    embedding_function=embeddings
)


# -------------------------
# 4. Create retriever
# -------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# -------------------------
# 5. Retrieval function
# -------------------------

def retrieve(query):

    results = retriever.invoke(query)

    return results
if __name__ == "__main__":

    query = "What are the limitations of medical RAG systems?"

    results = retrieve(query)

    print("\nRetrieved chunks:", len(results))

    for i, document in enumerate(results, start=1):

        page = document.metadata.get("page", "Unknown")

        print(f"\n--- Result {i} | Page {page + 1} ---")
        print(document.page_content)
