import ollama
import os
from typing import List, Tuple

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def generate_rag_answer(
    question: str,
    relevant_chunks: List[Tuple[str, float]],
    model: str = "llama3.2"
) -> dict:
    """
    Generate an answer using retrieved context chunks.
    This is the core RAG generation step.
    """

    # Build context from retrieved chunks
    context_parts = []
    for i, (chunk, score) in enumerate(relevant_chunks):
        context_parts.append(f"[Source {i + 1}]:\n{chunk}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a precise document analysis assistant. Answer the 
question below using ONLY the context provided. 

If the answer is not found in the context, say: 
"I could not find a direct answer to this question in the document."

Do not make up information. Be concise and factual.

Context from document:
{context}

Question: {question}

Answer:"""

    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1, "num_predict": 512}
    )

    answer = response['message']['content'].strip()

    # Return answer with source references
    return {
        "answer": answer,
        "sources": [
            {
                "chunk_index": i + 1,
                "text": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                "relevance_score": round(score, 4)
            }
            for i, (chunk, score) in enumerate(relevant_chunks)
        ],
        "model": model,
        "context_chunks_used": len(relevant_chunks)
    }