import ollama

from retrieve_langchain import retrieve


# -------------------------
# Ask a question
# -------------------------

question = input("\nAsk a question about the medical PDF: ")


# -------------------------
# Retrieve relevant chunks
# -------------------------

retrieved_chunks = retrieve(question)


# -------------------------
# Build context
# -------------------------

context_parts = []

for i, document in enumerate(retrieved_chunks, start=1):

    page = document.metadata.get("page", "Unknown")

    context_parts.append(
        f"Source {i} (PDF Page {page + 1}):\n"
        f"{document.page_content}"
    )


context = "\n\n".join(context_parts)


# -------------------------
# Build RAG prompt
# -------------------------

prompt = f"""
You answer questions using the provided medical document.

IMPORTANT RULES:
1. Use the CONTEXT below to answer the QUESTION.
2. Do not use outside knowledge.
3. Do not say the information is missing if the context contains the answer.
4. Give a concise answer.
5. Mention the relevant PDF page number(s).

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


# -------------------------
# Generate answer
# -------------------------

response = ollama.generate(
    model="llama3.2:1b",
    prompt=prompt
)


# -------------------------
# Display answer
# -------------------------

print("\n--- Answer ---")
print(response["response"])


# -------------------------
# Display sources
# -------------------------

print("\n--- Sources ---")

for i, document in enumerate(retrieved_chunks, start=1):

    page = document.metadata.get("page", "Unknown")

    print(f"Source {i}: PDF Page {page + 1}")
