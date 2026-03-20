# RAG with Your Own Data + Local Docker Model Runner

This project demonstrates an end-to-end **Retrieval-Augmented Generation (RAG)** workflow using your own documents, Chroma vector storage, Gemini embeddings, and a **local Docker-served LLM** for offline-style inference.

It is organized as a practical notebook journey so you can learn each stage step by step, then run question answering against your private knowledge base.

---

## What This Project Covers

### Core Features

- Build a RAG pipeline from scratch using your own data.
- Ingest and chunk both plain text and PDF documents.
- Generate embeddings with `models/gemini-embedding-2-preview`.
- Persist vectors in a local Chroma database (`chroma_db/`).
- Perform similarity search and retrieval over your indexed data.
- Run an interactive retrieval + answer workflow.
- Switch generation from cloud to a **local Docker Model Runner** endpoint using an OpenAI-compatible API (`http://localhost:12434/engines/v1`).

### Why This Setup Is Useful

- Your source documents remain in your own environment.
- Vector database is stored locally and reusable across notebooks.
- Local model serving reduces dependency on external generation APIs for inference.
- The notebook flow makes experimentation and learning easier.

---

## Project Structure

```text
rag-implementation-with-own-data/
├── README.md
├── pyproject.toml
├── requirements.txt
├── main.py
├── src/
│   ├── 1.0_RAG_With_Own_Text.ipynb
│   ├── 2.0_RAG_With_PDF.ipynb
│   ├── 3.0_Simalirity_Search_VectorDB.ipynb
│   └── 4.0_Docker_Model_Runner.ipynb
└── chroma_db/
    └── ... (persisted vector index)
```

---

## Tech Stack

- **Python**
- **LangChain** (`langchain`, `langchain-classic`, `langchain-community`)
- **ChromaDB** (`chromadb`, `langchain-chroma`)
- **Google GenAI** (`google-genai`, `langchain-google-genai`) for embedding and cloud LLM notebooks
- **OpenAI-compatible client** (`langchain-openai`) for local Docker model endpoint
- **PyPDF** for PDF ingestion
- **Jupyter notebooks** for step-by-step implementation

---

## End-to-End Workflow (What We Did)

Follow the notebooks in order:

### 1) `1.0_RAG_With_Own_Text.ipynb`

- Loaded environment configuration.
- Initialized Gemini chat model for early testing.
- Created `Document` objects from custom text data.
- Split text into chunks using `RecursiveCharacterTextSplitter`.
- Embedded chunks using `GoogleGenerativeAIEmbeddings`.
- Saved embeddings to local Chroma (`../chroma_db`).

### 2) `2.0_RAG_With_PDF.ipynb`

- Loaded PDFs with `PyPDFDirectoryLoader`.
- Split documents into retrieval-ready chunks.
- Embedded and stored chunks in the same Chroma database.
- Built retrieval + generation chain with a prompt template.
- Queried the indexed PDF knowledge through RAG.

### 3) `3.0_Simalirity_Search_VectorDB.ipynb`

- Reopened the persisted Chroma vector store.
- Performed similarity-based retrieval.
- Connected retriever with a Gemini chat model.
- Added interactive query handling via widgets for easier testing.

### 4) `4.0_Docker_Model_Runner.ipynb`

- Reused the **same embedding model and same Chroma store**.
- Replaced cloud generation model with local Docker-served model:
  - Model: `ai/qwen3:0.6B-F16`
  - Base URL: `http://localhost:12434/engines/v1`
- Built retrieval chain exactly as before.
- Answered questions using your indexed data with a locally served model backend.

---

## Setup Instructions

## 1. Clone the repository

```bash
git clone <your-repo-url>
cd rag-implementation-with-own-data
```

## 2. Create and activate a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

Use either `requirements.txt` or `pyproject.toml` workflow.

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

> Even when you use a local Docker-served LLM for generation, this project still uses Gemini embeddings for vector creation/loading in current notebooks.

---

## Running the Notebooks

Start Jupyter and execute notebooks in this order:

1. `src/1.0_RAG_With_Own_Text.ipynb`
2. `src/2.0_RAG_With_PDF.ipynb`
3. `src/3.0_Simalirity_Search_VectorDB.ipynb`
4. `src/4.0_Docker_Model_Runner.ipynb`

```bash
jupyter notebook
```

---

## Docker Model Runner Notes

For the local model notebook (`4.0_Docker_Model_Runner.ipynb`) to work, ensure:

- A Docker-based model runner is running locally.
- It exposes an OpenAI-compatible endpoint at:
  - `http://localhost:12434/engines/v1`
- The selected model (`ai/qwen3:0.6B-F16`) is available in that runner.

If the service is not running, retrieval chain setup will succeed, but answer generation calls will fail.

---

## Data and Privacy Considerations

- Your vector index is stored locally in `chroma_db/`.
- Only data you load gets indexed.
- Keep API keys private and never commit `.env`.
- Review model and API usage policies before production use.

---

## Current Limitations

- Main implementation is notebook-based (not yet packaged as production modules).
- Embeddings currently depend on Gemini API.
- Local model quality and latency depend on your hardware/model size.
- The repository does not include Docker compose files; it assumes a running local model runner endpoint.

---

## Next Improvements

- Add scripted CLI pipeline (`ingest`, `query`, `evaluate`).
- Add full local embedding option for fully offline operation.
- Add Docker Compose for one-command local startup.
- Add automated tests and benchmark notebook outputs.

---

## Quick Summary

This project shows how to:

1. Build a local vector knowledge base from your own text/PDF files.
2. Retrieve relevant context with similarity search.
3. Generate final answers with either cloud or local models.
4. Use Docker-served local models while keeping the same retrieval architecture.

