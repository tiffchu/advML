"""
Step 2: BM25 Keyword-Based Retrieval
=====================================
Uses rank_bm25 with simple whitespace + lowercase tokenization.
Persists the tokenized corpus and BM25 index via pickle.

Input:  data/raw/stratified_sample.parquet
Output: data/processed/chunks.parquet
        data/processed/tokenized_corpus.pkl
        data/processed/bm25_index.pkl
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


import pickle
import pandas as pd
from rank_bm25 import BM25Okapi

from src.utils import build_chunks, tokenize

#  Paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_FILE = RAW_DIR / "stratified_sample.parquet"  # input
CHUNKS_FILE = PROCESSED_DIR / "chunks.parquet"  # output
TOKENIZED_FILE = PROCESSED_DIR / "tokenized_corpus.pkl"
BM25_INDEX_FILE = PROCESSED_DIR / "bm25_index.pkl"


# 3. Build & persist BM25 index
def build_bm25_index(chunks: pd.DataFrame) -> tuple[BM25Okapi, list[list[str]]]:
    """Tokenize corpus and fit BM25Okapi index."""
    print(f"Tokenizing {len(chunks):,} chunks …")
    tokenized_corpus = [tokenize(text) for text in chunks["chunk_text"]]
    print("Building BM25 index …")
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus


def save_artefacts(
    chunks: pd.DataFrame,
    bm25: BM25Okapi,
    tokenized_corpus: list[list[str]],
) -> None:
    chunks.to_parquet(CHUNKS_FILE, index=False)
    print(f"Chunks saved            → {CHUNKS_FILE}")

    with open(TOKENIZED_FILE, "wb") as f:
        pickle.dump(tokenized_corpus, f)
    print(f"Tokenized corpus saved  → {TOKENIZED_FILE}")

    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved        → {BM25_INDEX_FILE}")


def load_artefacts() -> tuple[BM25Okapi, list[list[str]], pd.DataFrame]:
    with open(BM25_INDEX_FILE, "rb") as f:
        bm25 = pickle.load(f)
    with open(TOKENIZED_FILE, "rb") as f:
        tokenized_corpus = pickle.load(f)
    chunks = pd.read_parquet(CHUNKS_FILE)
    print("Loaded BM25 artefacts from disk.")
    return bm25, tokenized_corpus, chunks


# 4. Retrieval
def bm25_search(
    query: str,
    bm25: BM25Okapi,
    chunks: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """Return top-k chunks as a DataFrame with a 'score' column."""
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    results = chunks.copy()
    results["score"] = scores
    return (
        results.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    )


#  5. Main
def main():
    all_exist = (
        BM25_INDEX_FILE.exists() and TOKENIZED_FILE.exists() and CHUNKS_FILE.exists()
    )
    if all_exist:
        bm25, tokenized_corpus, chunks = load_artefacts()
    else:
        chunks = build_chunks(SAMPLE_FILE)
        bm25, tokenized_corpus = build_bm25_index(chunks)
        save_artefacts(chunks, bm25, tokenized_corpus)

    print(f"\nIndex ready — {len(tokenized_corpus):,} chunks indexed.\n")

    test_query = "washing machine not draining water"
    print(f"Query: '{test_query}'")
    print("─" * 60)
    results = bm25_search(test_query, bm25, chunks, top_k=5)

    for _, row in results.iterrows():
        print(
            f"score={row['score']:.4f}  parent_asin={row['parent_asin']}  rating={row['rating']}"
        )
        print(f"  {row['chunk_text'][:180].strip()} …\n")


if __name__ == "__main__":
    main()
