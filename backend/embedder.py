from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple

# Load a lightweight local embedding model
# all-MiniLM-L6-v2 is 80MB, fast, and strong for semantic search
MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def get_model() -> SentenceTransformer:
    """Lazy load the embedding model."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_chunks(chunks: List[str]) -> np.ndarray:
    """Convert text chunks into vector embeddings."""
    model = get_model()
    embeddings = model.encode(chunks, show_progress_bar=False)
    return np.array(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build a FAISS index for fast similarity search.
    IndexFlatL2 performs exact L2 distance search.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def retrieve_relevant_chunks(
    query: str,
    chunks: List[str],
    index: faiss.IndexFlatL2,
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Embed the query and retrieve the most relevant chunks.
    Returns list of (chunk_text, distance_score) tuples.
    """
    model = get_model()
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype="float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            results.append((chunks[idx], float(dist)))

    return results