from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdf_parser import extract_text_from_pdf, chunk_text
from embedder import embed_chunks, build_faiss_index, retrieve_relevant_chunks
from rag_engine import generate_rag_answer
import sys
import os

# Also import extraction from pdf-extractor pattern
sys.path.insert(0, os.path.dirname(__file__))

app = FastAPI(
    title="RAG Document Assistant API",
    description="Retrieval-Augmented Generation for document question answering using local LLM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory document store
# In production this would be a persistent vector database
document_store = {}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "llama3.2",
        "embedding_model": "all-MiniLM-L6-v2",
        "deployment": "local",
        "documents_indexed": len(document_store)
    }


@app.post("/index")
async def index_document(file: UploadFile = File(...)):
    """
    Upload and index a PDF document.
    Chunks the text, creates embeddings, and stores in FAISS.
    The document can then be queried repeatedly without re-indexing.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        file_bytes = await file.read()
        text = extract_text_from_pdf(file_bytes)

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text extracted. PDF may be image-based."
            )

        # Chunk and embed
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        embeddings = embed_chunks(chunks)
        index = build_faiss_index(embeddings)

        # Store in memory with filename as key
        doc_id = file.filename.replace(".pdf", "").replace(" ", "_")
        document_store[doc_id] = {
            "filename": file.filename,
            "chunks": chunks,
            "index": index,
            "total_chunks": len(chunks),
            "character_count": len(text)
        }

        return {
            "success": True,
            "doc_id": doc_id,
            "filename": file.filename,
            "total_chunks": len(chunks),
            "character_count": len(text),
            "message": f"Document indexed successfully. Ready for querying."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.post("/query")
async def query_document(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    """
    Upload a PDF and ask a question about it.
    Performs full RAG pipeline: extract, chunk, embed, retrieve, generate.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Extract and chunk
        file_bytes = await file.read()
        text = extract_text_from_pdf(file_bytes)

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text extracted. PDF may be image-based."
            )

        chunks = chunk_text(text, chunk_size=500, overlap=50)
        embeddings = embed_chunks(chunks)
        index = build_faiss_index(embeddings)

        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(
            query=question,
            chunks=chunks,
            index=index,
            top_k=3
        )

        # Generate answer
        result = generate_rag_answer(question, relevant_chunks)

        return {
            "success": True,
            "filename": file.filename,
            "question": question,
            "total_chunks_in_document": len(chunks),
            **result
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/query-indexed")
async def query_indexed_document(
    doc_id: str = Form(...),
    question: str = Form(...)
):
    """
    Query a previously indexed document by doc_id.
    Faster than /query as embedding step is skipped.
    """
    if doc_id not in document_store:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_id}' not found. Index it first using /index."
        )

    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        doc = document_store[doc_id]
        relevant_chunks = retrieve_relevant_chunks(
            query=question,
            chunks=doc["chunks"],
            index=doc["index"],
            top_k=3
        )

        result = generate_rag_answer(question, relevant_chunks)

        return {
            "success": True,
            "doc_id": doc_id,
            "filename": doc["filename"],
            "question": question,
            **result
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/indexed-documents")
def list_indexed_documents():
    """List all currently indexed documents."""
    return {
        "total": len(document_store),
        "documents": [
            {
                "doc_id": doc_id,
                "filename": doc["filename"],
                "total_chunks": doc["total_chunks"],
                "character_count": doc["character_count"]
            }
            for doc_id, doc in document_store.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)