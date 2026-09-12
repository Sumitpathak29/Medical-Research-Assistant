from pathlib import Path

import gradio as gr
import ollama

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------
# 1. Project paths
# -------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_DIR / "chroma_db"


# -------------------------
# 2. Embedding model
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------
# 3. Persistent vector store
# -------------------------

vectorstore = Chroma(
    collection_name="medical_documents",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PATH)
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)


# -------------------------
# 4. Process uploaded PDFs
# -------------------------

def process_pdf(pdf_files):

    if not pdf_files:
        return "Please upload at least one PDF."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    status_messages = []

    # Handle both a single file and multiple files
    if isinstance(pdf_files, str):
        pdf_files = [pdf_files]

    for pdf_file in pdf_files:

        filename = Path(pdf_file).name

        # -------------------------
        # Check if document exists
        # -------------------------

        existing = vectorstore.get(
            where={"source": filename}
        )

        if existing["ids"]:

            status_messages.append(
                f"'{filename}' is already in the knowledge base "
                f"({len(existing['ids'])} chunks)."
            )

            continue

        # -------------------------
        # Load PDF
        # -------------------------

        loader = PyPDFLoader(pdf_file)
        documents = loader.load()

        # -------------------------
        # Split into chunks
        # -------------------------

        chunks = splitter.split_documents(documents)

        # -------------------------
        # Add metadata
        # -------------------------

        for chunk in chunks:
            chunk.metadata["source"] = filename

        # -------------------------
        # Create unique IDs
        # -------------------------

        document_ids = []

        for i, chunk in enumerate(chunks):

            page = chunk.metadata.get("page", 0)

            chunk_id = (
                f"{filename}-page-{page}-chunk-{i}"
            )

            document_ids.append(chunk_id)

        # -------------------------
        # Add chunks to ChromaDB
        # -------------------------

        vectorstore.add_documents(
            documents=chunks,
            ids=document_ids
        )

        status_messages.append(
            f"'{filename}' added successfully "
            f"({len(documents)} pages, {len(chunks)} chunks)."
        )

    return "\n\n".join(status_messages)


# -------------------------
# 5. Ask question
# -------------------------

def ask_question(question):

    if not question or not question.strip():

        return (
            "Please enter a question.",
            ""
        )

    # -------------------------
    # Retrieve relevant chunks
    # -------------------------

    results = retriever.invoke(question)

    if not results:

        return (
            "I could not find relevant information "
            "in the uploaded documents.",
            ""
        )

    # -------------------------
    # Build context with sources
    # -------------------------

    context_parts = []

    for i, document in enumerate(results, start=1):

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        filename = document.metadata.get(
            "source",
            "Unknown document"
        )

        if isinstance(page, int):

            page_number = page + 1

        else:

            page_number = "Unknown"

        context_parts.append(
            f"Source {i} "
            f"({filename}, PDF Page {page_number}):\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    # -------------------------
    # RAG prompt
    # -------------------------

    prompt = f"""
You are a medical research assistant.

Answer the user's question using ONLY the information
provided in the CONTEXT below.

IMPORTANT RULES:

1. Use only the provided context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. If the context does not contain enough information,
   say:
   "I could not find this information in the uploaded documents."
5. Give a concise and clear answer.
6. When possible, mention the document name and PDF page
   where the information was found.
7. If information comes from multiple documents, clearly
   distinguish between them.

CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""

    # -------------------------
    # Generate answer with Ollama
    # -------------------------

    response = ollama.generate(
        model="llama3.2:1b",
        prompt=prompt
    )

    answer_text = response["response"]

    # -------------------------
    # Build source list
    # -------------------------

    sources = []

    for i, document in enumerate(results, start=1):

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        filename = document.metadata.get(
            "source",
            "Unknown document"
        )

        if isinstance(page, int):

            sources.append(
                f"Source {i}: "
                f"{filename} - PDF Page {page + 1}"
            )

        else:

            sources.append(
                f"Source {i}: "
                f"{filename} - Page unavailable"
            )

    source_text = "\n".join(sources)

    return answer_text, source_text


# -------------------------
# 6. Gradio interface
# -------------------------

with gr.Blocks() as demo:

    gr.Markdown(
        "# 🩺 Medical Research Assistant"
    )

    gr.Markdown(
        "Upload multiple medical PDFs and ask questions "
        "across the entire knowledge base."
    )

    # -------------------------
    # PDF section
    # -------------------------

    pdf = gr.File(
        label="Upload Medical PDFs",
        file_types=[".pdf"],
        type="filepath",
        file_count="multiple"
    )

    process_button = gr.Button(
        "Add PDFs to Knowledge Base"
    )

    status = gr.Textbox(
        label="Processing Status",
        lines=8
    )

    # -------------------------
    # Question section
    # -------------------------

    question = gr.Textbox(
        label="Ask a question",
        placeholder=(
            "Ask something about your uploaded "
            "medical documents..."
        )
    )

    ask_button = gr.Button(
        "Ask Question"
    )

    # -------------------------
    # Answer section
    # -------------------------

    answer = gr.Textbox(
        label="Answer",
        lines=8
    )

    sources = gr.Textbox(
        label="Sources",
        lines=5
    )

    # -------------------------
    # Button actions
    # -------------------------

    process_button.click(
        fn=process_pdf,
        inputs=pdf,
        outputs=status
    )

    ask_button.click(
        fn=ask_question,
        inputs=question,
        outputs=[answer, sources]
    )


# -------------------------
# 7. Launch
# -------------------------

if __name__ == "__main__":
    demo.launch()
