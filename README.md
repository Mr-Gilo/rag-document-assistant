# RAG Document Assistant

A locally hosted Retrieval-Augmented Generation (RAG) system for 
document question answering. Ask questions about any PDF in natural 
language and receive grounded answers with source citations.

All processing happens on your machine. No data leaves your device.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![FAISS](https://img.shields.io/badge/FAISS-1.8.0-orange)
![Ollama](https://img.shields.io/badge/Ollama-llama3.2-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)

## Architecture

PDF → PyMuPDF → Text Chunks (500 words, 50 overlap)

↓

SentenceTransformer (all-MiniLM-L6-v2)

↓

FAISS Vector Index

↓

Query → Semantic Similarity Search

↓

Top-K Relevant Chunks Retrieved

↓

Ollama (llama3.2) + Context Prompt

↓

Grounded Answer + Source Citations

## Features

- Local LLM inference via Ollama — no API keys required
- Local embeddings via SentenceTransformers — no cloud dependency
- FAISS vector search for fast semantic retrieval
- Source citations showing exactly which document chunks were used
- Two query modes: direct upload or pre-indexed for repeated use
- REST API with full Swagger documentation
- Streamlit chat-style interface

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Ollama (llama3.2) |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS (IndexFlatL2) |
| Backend | FastAPI + Uvicorn |
| PDF Parsing | PyMuPDF (fitz) |
| Frontend | Streamlit |

## Prerequisites

- Python 3.11+
- Ollama installed: https://ollama.com/download
- llama3.2 model: `ollama pull llama3.2`

## Installation

```bash
git clone https://github.com/Mr-Gilo/rag-document-assistant.git
cd rag-document-assistant

conda create -n rag-assistant python=3.11 -y
conda activate rag-assistant

pip install -r backend/requirements.txt
pip install -r requirements.txt
```

## Running the Application

**Terminal 1 — Backend:**
```bash
conda activate rag-assistant
cd backend
python main.py
```
Backend runs at http://localhost:8001
API docs at http://localhost:8001/docs

**Terminal 2 — Frontend:**
```bash
conda activate rag-assistant
streamlit run app.py
```
Frontend runs at http://localhost:8501

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | System status and model info |
| POST | /index | Index a PDF for repeated querying |
| POST | /query | Upload PDF and ask a question |
| POST | /query-indexed | Query a previously indexed document |
| GET | /indexed-documents | List all indexed documents |

## Example

Upload a motor insurance claim PDF and ask:

- "What is the date of the incident?"
- "Who are the parties involved?"
- "What damages are being claimed?"
- "Did the third party admit liability?"

The system retrieves the most relevant sections and generates 
a grounded answer citing its sources.

## Related Project

This project extends the PDF extraction patterns from 
[pdf-extractor](https://github.com/Mr-Gilo/pdf-extractor), 
adding semantic search and retrieval-augmented generation.

## Roadmap

- [x] PDF text extraction and chunking
- [x] Local embeddings via SentenceTransformers
- [x] FAISS vector index for semantic search
- [x] RAG generation with source citations
- [x] Document pre-indexing for repeated queries
- [x] FastAPI REST backend
- [x] Streamlit frontend
- [ ] Docker containerisation
- [ ] Persistent vector store (ChromaDB)
- [ ] Multi-document cross-querying
- [ ] Confidence score filtering