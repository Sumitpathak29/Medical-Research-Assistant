# 🩺 Medical Research Assistant

A **multi-document medical Retrieval-Augmented Generation (RAG) system** that allows users to upload medical PDFs and ask questions about their contents.

The system processes documents, generates semantic embeddings, stores them in **ChromaDB**, retrieves relevant chunks for a user's question, and uses **Llama 3.2** through **Ollama** to generate grounded answers based only on the retrieved context.

The project was initially implemented using core Python libraries to understand the underlying RAG pipeline and was later extended using **LangChain** and a **Gradio** interface for the final application.

> ⚠️ **Disclaimer:** This project is intended for research and educational purposes. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice.

---

## 🚀 Features

* 📄 Upload and process medical PDF documents
* 📚 Support for **multiple documents** in a shared knowledge base
* ✂️ Automatic document chunking
* 🧠 Semantic embeddings using Sentence Transformers
* 🗄️ Persistent vector storage with ChromaDB
* 🔎 Similarity-based retrieval of relevant document chunks
* 🦙 Local LLM inference using Llama 3.2 through Ollama
* 📑 Source and PDF page attribution
* 🔗 LangChain-based retrieval pipeline
* 🖥️ Gradio web interface
* 🎯 LoRA/QLoRA fine-tuning experiment for domain adaptation
* 🔐 Runs locally without requiring a paid LLM API

---

## 🏗️ Architecture

```text
                  Medical PDFs
                       │
                       ▼
              PDF Text Extraction
                       │
                       ▼
                 Text Chunking
                       │
                       ▼
             Sentence Transformers
                  Embeddings
                       │
                       ▼
                  ChromaDB
              Vector Knowledge Base
                       │
                       │
               User Question
                       │
                       ▼
              Query Embedding
                       │
                       ▼
             Similarity Retrieval
                       │
                       ▼
             Relevant Text Chunks
                       │
                       ▼
                 RAG Prompt
                       │
                       ▼
             Llama 3.2 via Ollama
                       │
                       ▼
              Grounded Answer
                       │
                       ▼
             Sources + Page Numbers
```

---

## 🔬 RAG Pipeline

The application follows the standard Retrieval-Augmented Generation workflow:

### 1. Document Ingestion

Medical PDFs are loaded and their text is extracted.

### 2. Chunking

Large documents are divided into smaller overlapping chunks.

```text
Chunk Size:    1000 characters
Chunk Overlap: 200 characters
```

The overlap helps preserve context between neighboring chunks.

### 3. Embedding Generation

Each chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### 4. Vector Storage

The embeddings and corresponding document chunks are stored in **ChromaDB**.

### 5. Retrieval

When the user asks a question, the question is converted into an embedding and the most semantically similar chunks are retrieved.

The current application retrieves the top:

```text
k = 5
```

relevant chunks.

### 6. Generation

The retrieved chunks are inserted into a prompt and passed to:

```text
Llama 3.2 1B
        ↓
Ollama
```

The model is instructed to answer using only the retrieved context.

### 7. Source Attribution

Retrieved documents include metadata such as:

* PDF filename
* PDF page number
* Retrieved source number

This allows the generated answer to be traced back to the source material.

---

## 🧪 Manual RAG vs LangChain

The project contains both a manual implementation and a LangChain implementation.

### Manual implementation

The initial version was built using:

```text
PyPDF
Sentence Transformers
ChromaDB
Ollama
```

This implementation demonstrates the underlying mechanics of a RAG system without relying heavily on abstraction frameworks.

### LangChain implementation

The pipeline was then implemented using:

```text
PyPDFLoader
RecursiveCharacterTextSplitter
HuggingFaceEmbeddings
Chroma
LangChain Retriever
Ollama
```

This version demonstrates how the same architecture can be implemented using modern LLM tooling.

---

## 📁 Project Structure

```text
Medical-Research-Assistant/
│
├── src/
│   ├── ingest.py
│   ├── retrieve.py
│   ├── rag.py
│   ├── ingestion_langchain.py
│   ├── retrieve_langchain.py
│   ├── generate.py
│   └── app.py
│
├── notebooks/
│   └── Medical_Research_Assistant.ipynb
│
├── data/
│   └── documents/
│
├── README.md
├── requirements.txt
└── .gitignore
```

### Source files

| File                     | Description                                                     |
| ------------------------ | --------------------------------------------------------------- |
| `ingest.py`              | Manual PDF ingestion, chunking, embeddings and ChromaDB storage |
| `retrieve.py`            | Manual semantic retrieval from ChromaDB                         |
| `rag.py`                 | Manual end-to-end RAG pipeline using Ollama                     |
| `ingestion_langchain.py` | PDF ingestion using LangChain components                        |
| `retrieve_langchain.py`  | LangChain-based document retrieval                              |
| `generate.py`            | Generates answers from retrieved context using Ollama           |
| `app.py`                 | Final multi-document Gradio application                         |

---

## 🖥️ Final Application

The final application provides a simple interface where users can:

1. Upload a medical PDF.
2. Add it to the knowledge base.
3. Upload additional PDFs.
4. Ask questions across the uploaded documents.
5. Receive an answer generated by the local LLM.
6. View the source documents and PDF pages used for retrieval.

---

## 🛠️ Tech Stack

### Programming

* Python

### RAG / LLM

* Llama 3.2
* Ollama
* Retrieval-Augmented Generation (RAG)

### Embeddings

* Sentence Transformers
* `all-MiniLM-L6-v2`

### Vector Database

* ChromaDB

### Frameworks

* LangChain
* Hugging Face Transformers
* PEFT
* TRL

### Fine-Tuning

* LoRA
* QLoRA
* Parameter-Efficient Fine-Tuning

### Interface

* Gradio

### Document Processing

* PyPDF
* LangChain PyPDFLoader

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Medical-Research-Assistant
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🦙 Ollama Setup

Install Ollama and pull the model:

```bash
ollama pull llama3.2:1b
```

Make sure Ollama is running before starting the application.

You can verify the model with:

```bash
ollama list
```

---

## ▶️ Running the Application

Start the Gradio application:

```bash
python src/app.py
```

Gradio will provide a local URL similar to:

```text
http://127.0.0.1:7860
```

Open the URL in your browser.

---

## 📄 Adding Documents

Place any sample medical PDFs inside:

```text
data/documents/
```

For the final multi-document application, PDFs can also be uploaded through the Gradio interface and added to the shared ChromaDB knowledge base.

Example:

```text
data/
└── documents/
    ├── diabetes.pdf
    ├── medical_rag.pdf
    └── cardiovascular_research.pdf
```

---

## 🎯 Fine-Tuning Experiment

The project also includes an experimental **LoRA/QLoRA fine-tuning workflow** using Hugging Face Transformers, PEFT and TRL.

The purpose of the experiment is to explore how a language model can be adapted toward a specific domain using parameter-efficient fine-tuning rather than updating all model parameters.

The fine-tuning workflow is maintained separately from the main RAG application.

```text
Base Llama Model
       ↓
4-bit Quantization
       ↓
LoRA Adapters
       ↓
Domain Fine-Tuning
       ↓
Fine-Tuned Model
```

The main application continues to use the local Ollama-based model for inference.

---

## 📊 Limitations

* The quality of answers depends heavily on retrieval quality.
* Poorly extracted or poorly structured PDFs can reduce retrieval accuracy.
* The small local LLM has limited reasoning capabilities compared with larger frontier models.
* The system does not independently verify medical claims.
* Semantic retrieval can miss information when relevant content is distributed across many sections.
* The application should not be used for clinical decision-making.

---

## 🔮 Future Improvements

Potential improvements include:

* Better document parsing for tables and structured medical content
* Hybrid keyword + semantic retrieval
* Reranking retrieved documents
* Query expansion
* Conversational memory
* Improved citation handling
* Evaluation using RAG-specific metrics
* Larger or domain-specialized embedding models
* Larger local LLMs
* Automated hallucination and faithfulness evaluation
* Improved multi-document filtering and document management
* Deployment as a production web application

---

## 📚 Learning Objectives

This project was built to understand the complete lifecycle of a RAG application:

```text
Document Processing
        ↓
Chunking
        ↓
Embeddings
        ↓
Vector Database
        ↓
Semantic Retrieval
        ↓
Prompt Construction
        ↓
LLM Generation
        ↓
Source Attribution
        ↓
Application Interface
```

It also provides hands-on experience with **LangChain, vector databases, local LLM inference, multi-document retrieval, and parameter-efficient fine-tuning**.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.

It does not provide medical diagnosis, treatment recommendations, or professional medical advice. Always consult a qualified healthcare professional for medical decisions.
