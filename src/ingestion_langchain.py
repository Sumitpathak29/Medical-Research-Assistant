from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# -------------------------
# 1. Paths
# -------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = PROJECT_DIR / "data" / "documents" / "medical.pdf"
CHROMA_PATH = PROJECT_DIR / "chroma_db"


# -------------------------
# 2. Load PDF
# -------------------------

loader = PyPDFLoader(str(PDF_PATH))

documents = loader.load()

print("Pages loaded:", len(documents))


# -------------------------
# 3. Split into chunks
# -------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Chunks created:", len(chunks))


# -------------------------
# 4. Create embeddings
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------
# 5. Store in ChromaDB
# -------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(CHROMA_PATH),
    collection_name="medical_documents"
)

print("Documents stored in ChromaDB:", vectorstore._collection.count())
