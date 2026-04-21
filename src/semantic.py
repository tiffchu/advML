"""
Step 3: Semantic Retrieval
===========================
Generates embeddings with sentence-transformers (all-MiniLM-L6-v2)
and indexes them with raw FAISS for fast nearest-neighbour search.

Input:  data/processed/chunks.parquet   (built by src/bm25.py)
Output: data/processed/embeddings.npy
        data/processed/faiss.index
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.utils import load_chunks

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_FILE = PROCESSED_DIR / "chunks.parquet"  # written by bm25.py
EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings.npy"
FAISS_INDEX_FILE = PROCESSED_DIR / "faiss.index"

MODEL_NAME = "all-MiniLM-L6-v2"


# embed corpus
def build_embeddings(chunks: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    """
    Encode all chunk_text strings into L2-normalised float32 embeddings.
    L2 normalisation means inner product == cosine similarity.
    """
    texts = chunks["chunk_text"].tolist()
    print(f"Encoding {len(texts):,} chunks with '{MODEL_NAME}' …")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    faiss.normalize_L2(embeddings)
    return embeddings


# build FAISS index
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    IndexFlatIP = exact inner-product search (cosine similarity after normalisation).
    Fine for ~1k docs; swap for IndexIVFFlat if scaling to millions.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built — {index.ntotal:,} vectors, dim={dim}.")
    return index


# persist
def save_artefacts(embeddings: np.ndarray, index: faiss.IndexFlatIP) -> None:
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"Embeddings saved  → {EMBEDDINGS_FILE}")
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    print(f"FAISS index saved → {FAISS_INDEX_FILE}")


def load_artefacts() -> tuple[faiss.IndexFlatIP, np.ndarray, pd.DataFrame]:
    index = faiss.read_index(str(FAISS_INDEX_FILE))
    embeddings = np.load(EMBEDDINGS_FILE)
    chunks = load_chunks(CHUNKS_FILE)  # always from processed/
    print(f"Loaded FAISS index ({index.ntotal:,} vectors) from disk.")
    return index, embeddings, chunks


# retrieval
def semantic_search(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: pd.DataFrame,
    model: SentenceTransformer,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Encode the query, search the FAISS index, return top-k chunks
    as a DataFrame with a 'score' column (cosine similarity, 0–1).
    """
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, top_k)
    results = chunks.iloc[indices[0]].copy()
    results = results.reset_index(drop=True)
    results["score"] = scores[0]
    return results


def main():
    # bm25.py must run first to produce chunks.parquet
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"{CHUNKS_FILE} not found — run src/bm25.py first to build chunks."
        )

    model = SentenceTransformer(MODEL_NAME)

    if FAISS_INDEX_FILE.exists() and EMBEDDINGS_FILE.exists():
        index, embeddings, chunks = load_artefacts()
    else:
        chunks = load_chunks(CHUNKS_FILE)
        embeddings = build_embeddings(chunks, model)
        index = build_faiss_index(embeddings)
        save_artefacts(embeddings, index)

    print(f"\nIndex ready — {index.ntotal:,} vectors indexed.\n")

    test_query = "washing machine not draining water"
    print(f"Query: '{test_query}'")
    print("─" * 60)
    results = semantic_search(test_query, index, chunks, model, top_k=5)

    for _, row in results.iterrows():
        print(
            f"score={row['score']:.4f}  parent_asin={row['parent_asin']}  rating={row['rating']}"
        )
        print(f"  {row['chunk_text'][:180].strip()} …\n")


if __name__ == "__main__":
    main()
