from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.bm25 import bm25_search, load_artefacts as load_bm25_artefacts
from src.semantic import (
    MODEL_NAME,
    semantic_search,
    load_artefacts as load_semantic_artefacts,
)

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_FILE = RAW_DIR / "stratified_sample.parquet"
CHUNKS_FILE = PROCESSED_DIR / "chunks.parquet"

st.set_page_config(page_title="Appliances Search", layout="wide")


@st.cache_data
def load_chunks_with_metadata() -> pd.DataFrame:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"{CHUNKS_FILE} not found. Run src/bm25.py first.")

    chunks = pd.read_parquet(CHUNKS_FILE).copy()
    chunks = chunks.reset_index(drop=True)

    if SAMPLE_FILE.exists():
        raw = pd.read_parquet(SAMPLE_FILE).reset_index(drop=True)
        raw.insert(0, "chunk_id", range(len(raw)))
        raw = raw[["chunk_id", "text", "product_title", "rating"]]
        chunks = chunks.merge(raw, on="chunk_id", how="left", suffixes=("", "_raw"))

        if "product_title_raw" in chunks.columns:
            chunks["product_title"] = chunks["product_title"].fillna(
                chunks["product_title_raw"]
            )
            chunks = chunks.drop(columns=["product_title_raw"])

        if "rating_raw" in chunks.columns:
            chunks["rating"] = chunks["rating"].fillna(chunks["rating_raw"])
            chunks = chunks.drop(columns=["rating_raw"])

    return chunks.sort_values("chunk_id").reset_index(drop=True)


@st.cache_resource
def load_bm25():
    bm25, _, _ = load_bm25_artefacts()
    return bm25


@st.cache_resource
def load_semantic():
    index, _, _ = load_semantic_artefacts()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    return index, model


@st.cache_resource
def load_rag():
    from src.rag_pipe import RAGPipeline

    return RAGPipeline()


@st.cache_resource
def load_hybrid():
    from src.hybrid import HybridRAGPipeline

    return HybridRAGPipeline()


def truncate(text: str, n: int = 200) -> str:
    if not isinstance(text, str):
        return ""
    text = " ".join(text.strip().split())
    return text if len(text) <= n else text[:n].rstrip() + "..."

#had chatgpt assist with displaying the data and help with streamlit syntax

def format_rating(rating) -> str:
    if pd.isna(rating):
        return "N/A"

    try:
        rating_value = float(rating)
    except (TypeError, ValueError):
        return str(rating)

    return f"{rating_value:.1f}/5"

def prepare_results_for_display(
    results: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    display = results.copy()

    needed_cols = {"chunk_id", "product_title", "rating", "chunk_text"}
    if "chunk_id" in display.columns and not needed_cols.issubset(display.columns):
        display = display.merge(
            metadata[["chunk_id", "product_title", "rating", "chunk_text"]],
            on="chunk_id",
            how="left",
            suffixes=("", "_meta"),
        )

        for col in ["product_title", "rating", "text"]:
            meta_col = f"{col}_meta"
            if meta_col in display.columns:
                display[col] = display[col].fillna(display[meta_col])
                display = display.drop(columns=[meta_col])

    return display.reset_index(drop=True)


def show_results(results: pd.DataFrame, score_col: str = "score"):
    if results.empty:
        st.info("No results found.")
        return

    for idx, row in results.iterrows():
        title = row.get("product_title") or "Unknown Product"
        review_text = row.get("text") or row.get("chunk_text") or ""
        rating = format_rating(row.get("rating"))
        score = row.get(score_col)

        with st.container(border=True):
            st.markdown(f"**{idx + 1}. {title}**")
            st.write(f"Rating: {rating}")

            if pd.notna(score):
                st.caption(f"Retrieval score: {float(score):.4f}")

            if review_text:
                st.write(truncate(review_text, 200))


def show_sources(results: pd.DataFrame, score_col: str):
    st.markdown("#### Sources")

    for idx, row in results.iterrows():
        title = row.get("product_title") or "Unknown Product"
        review_text = row.get("text") or row.get("chunk_text") or ""
        rating = format_rating(row.get("rating"))
        score = row.get(score_col)

        with st.container(border=True):
            st.markdown(f"**[{idx + 1}] {title}**")
            st.write(f"Rating: {rating}")

            if pd.notna(score):
                st.caption(f"Retrieval score: {float(score):.4f}")

            if review_text:
                st.write(truncate(review_text, 200))


st.title("Appliances Review Search")
st.caption("Search appliance reviews directly or switch to RAG mode for a better answer.")

chunks = load_chunks_with_metadata()

search_tab, rag_tab = st.tabs(["Search", "RAG"])

with search_tab:
    st.subheader("Search Only")
    st.caption("Milestone 1 retrieval")

    mode = st.radio(
        "Search Mode", ["BM25", "Semantic"], horizontal=True, key="search_mode"
    )
    query = st.text_input(
        "Enter your query", value="coffee maker quiet", key="search_query"
    )
    run_search = st.button("Search", type="primary", key="btn_search")

    if run_search and query:
        if mode == "BM25":
            bm25 = load_bm25()
            results = bm25_search(query, bm25, chunks, top_k=3)
        else:
            index, model = load_semantic()
            results = semantic_search(query, index, chunks, model, top_k=3)

        results = prepare_results_for_display(results, chunks)
        st.subheader("Top Results")
        show_results(results)

with rag_tab:
    st.subheader("RAG Mode")

    rag_mode = st.radio(
        "Retriever", ["Semantic RAG", "Hybrid RAG"], horizontal=True, key="rag_mode"
    )
    rag_query = st.text_input(
        "Enter your query",
        value="coffee maker quiet",
        key="rag_query",
    )
    run_rag = st.button("Ask", type="primary", key="btn_rag")

    if run_rag and rag_query:
        try:
            with st.spinner("Retrieving documents and generating answer..."):
                if rag_mode == "Semantic RAG":
                    pipeline = load_rag()
                    answer, docs = pipeline.run(rag_query, top_k=5)
                    score_col = "score"
                else:
                    pipeline = load_hybrid()
                    answer, docs = pipeline.run(rag_query, top_k=5)
                    score_col = "rrf_score"

            docs = prepare_results_for_display(docs, chunks)
            clean_answer = " ".join(answer.strip().split()) if isinstance(answer, str) else ""

            st.markdown("### Generated Answer")
            st.markdown(truncate(clean_answer, 700))
            if len(clean_answer) > 700:
                st.caption("Answer truncated for readability")

            show_sources(docs, score_col=score_col)

        except Exception as exc:
            st.error(
                "RAG mode could not complete so make sure retrieval artefacts exist and Ollama is running, and then try again"
            )
            st.exception(exc)
