from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_DIR / "data" / "documents" / "medical.pdf"
CHROMA_PATH = PROJECT_DIR / "chroma_db"


# -------------------------
# 1. Extract PDF text
# -------------------------

reader = PdfReader(PDF_PATH)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print("PDF:", PDF_PATH)
print("Pages:", len(reader.pages))
print("Characters extracted:", len(text))


# -------------------------
# 2. Create chunks
# -------------------------

chunk_size = 1000
chunk_overlap = 200

chunks = []

start = 0

while start < len(text):
    end = start + chunk_size

    chunk = text[start:end]

    if chunk.strip():
        chunks.append(chunk.strip())

    start += chunk_size - chunk_overlap


print("Chunks created:", len(chunks))
print("First chunk length:", len(chunks[0]))
print("Last chunk length:", len(chunks[-1]))


# -------------------------
# 3. Create embeddings
# -------------------------

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = embedding_model.encode(chunks)

print("Embeddings created:", len(embeddings))
print("Embedding dimensions:", embeddings.shape)


# -------------------------
# 4. Store in ChromaDB
# -------------------------

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_or_create_collection(
    name="medical_documents"
)

collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist()
)

print("ChromaDB documents:", collection.count())
print("ChromaDB location:", CHROMA_PATH)
