import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pickle
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.bm25 import bm25_search, BM25_INDEX_FILE, TOKENIZED_FILE, CHUNKS_FILE
from src.semantic import semantic_search, load_artefacts as load_semantic, MODEL_NAME
from src.rag_pipe import OllamaLLM, build_context, build_prompt

# had chatgpt review and edit my initial code and it suggested using a class so it doesnt have to reload the indexes and model from disk every time. I also asked it to help me fix and better implement my RRF function


class BM25Retriever:
    def __init__(self):
        with open(BM25_INDEX_FILE, "rb") as f:
            self.bm25 = pickle.load(f)
        self.chunks = pd.read_parquet(CHUNKS_FILE)

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        return bm25_search(query, self.bm25, self.chunks, top_k=top_k)


class SemanticRetriever:
    def __init__(self):
        self.index, self.embeddings, self.chunks = load_semantic()
        self.model = SentenceTransformer(MODEL_NAME)

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        return semantic_search(query, self.index, self.chunks, self.model, top_k=top_k)


def reciprocal_rank_fusion(
    results_list: list[pd.DataFrame],
    k: int = 60,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Merge multiple ranked result DataFrames using RRF
    k=60 : smoothing constant (commonly 60) from lecture notes
    """
    rrf_scores: dict[int, float] = {}
    chunk_rows: dict[int, pd.Series] = {}

    for results in results_list:
        for rank, (_, row) in enumerate(results.iterrows()):
            cid = int(row["chunk_id"])
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunk_rows[cid] = row  # keep the latest row for metadata

    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

    fused = pd.DataFrame([chunk_rows[cid] for cid in sorted_ids]).reset_index(drop=True)
    fused["rrf_score"] = [rrf_scores[cid] for cid in sorted_ids]
    return fused


class HybridRAGPipeline:
    def __init__(self, llm_model: str = "llama3.1:8b"):
        self.bm25_retriever = BM25Retriever()
        self.semantic_retriever = SemanticRetriever()

        # reuse from existing rag.py
        from src.rag_pipe import OllamaLLM, build_context, build_prompt

        self.llm = OllamaLLM(llm_model)
        self.build_context = build_context
        self.build_prompt = build_prompt

    def run(self, query: str, top_k: int = 5, prompt_version: int = 1):
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k)
        semantic_results = self.semantic_retriever.retrieve(query, top_k=top_k)

        fused_docs = reciprocal_rank_fusion(
            [bm25_results, semantic_results], top_k=top_k
        )

        context = self.build_context(fused_docs)
        prompt = self.build_prompt(query, context, prompt_version)
        answer = self.llm.generate(prompt)

        return answer, fused_docs


if __name__ == "__main__":
    pipeline = HybridRAGPipeline()

    query = "quiet indoor fan"
    answer, docs = pipeline.run(query, top_k=5)

    print("\nQUERY:", query)
    print("\nANSWER:\n", answer)
    print("\nFUSED DOCS:")
    for _, row in docs.iterrows():
        print(f"rrf={row['rrf_score']:.4f}  {row.get('product_title', 'N/A')}")
        print(f"{str(row.get('chunk_text', ''))[:160]} …\n")
