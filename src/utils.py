"""
Utilities for corpus construction and tokenization.
"""

from pathlib import Path

import pandas as pd


def tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase + whitespace split."""
    if not isinstance(text, str):
        return []
    return text.lower().split()


def build_chunks(path: Path) -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_parquet(path)

    df["chunk_text"] = (
        df["text"].fillna("") + " " + df["product_title"].fillna("")
    ).str.strip()

    chunks = df[[
        "parent_asin",
        "product_title",  
        "rating",
        "chunk_text"
    ]].copy()

    chunks.insert(0, "chunk_id", range(len(chunks)))

    print(f"Built {len(chunks):,} chunks.")
    return chunks


def load_chunks(path: Path) -> pd.DataFrame:
    """Load pre-built chunks parquet."""
    return pd.read_parquet(path)
