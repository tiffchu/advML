import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import ollama

from src.semantic import load_artefacts, semantic_search, MODEL_NAME


# asked chatgpt to correct my pipeline since we kept getting this output: top docs: None while still getting scores
class SemanticRetriever:
    def __init__(self):
        self.index, self.embeddings, self.chunks = load_artefacts()
        self.model = SentenceTransformer(MODEL_NAME)

    def retrieve(self, query: str, top_k: int = 5):  # we chose top_k as 5
        results = semantic_search(
            query, self.index, self.chunks, self.model, top_k=top_k
        )
        return results


def build_context(docs: pd.DataFrame) -> str:
    context_blocks = []

    for _, row in docs.iterrows():
        block = f"""
ASIN: {row.get('parent_asin', 'N/A')}
Rating: {row.get('rating', 'N/A')}/5
Review: {row.get('chunk_text', '')}
"""
        context_blocks.append(block.strip())

    return "\n\n".join(context_blocks)


# prompt template

SYSTEM_PROMPT_V1 = """
You are a helpful Amazon shopping assistant.
Answer the question using ONLY the provided context.
Always cite the Product ASIN when relevant.
Be concise. Do not make up information, do not hallucinate please.
"""

SYSTEM_PROMPT_V2 = """
You are an expert appliances analyst.
Use ONLY the given reviews and metadata to answer.
If the answer is not in the context, say "I don't know".
Focus on practical insights from customer reviews.
"""


def build_prompt(query: str, context: str, version: int = 1) -> str:
    system_prompt = SYSTEM_PROMPT_V1 if version == 1 else SYSTEM_PROMPT_V2

    return f"""{system_prompt}

CONTEXT:
{context}

QUESTION:
{query}

Answer based on the Amazon datasets: """


# LLM OLLAMA
class OllamaLLM:
    def __init__(self, model_name="llama3.1:8b"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]


# RAG PIPELINE
class RAGPipeline:
    def __init__(self, model_name="llama3.1:8b"):
        self.retriever = SemanticRetriever()
        self.llm = OllamaLLM(model_name)

    def run(self, query: str, top_k: int = 5, prompt_version: int = 1):
        docs = self.retriever.retrieve(query, top_k=top_k)

        context = build_context(docs)

        prompt = build_prompt(query, context, prompt_version)

        answer = self.llm.generate(prompt)

        return answer, docs


if __name__ == "__main__":
    rag = RAGPipeline()

    query = "washing machine not draining water"
    answer, docs = rag.run(query, top_k=5)

    print("\nQUERY:")
    print(query)

    print("\nANSWER")
    print(answer)

    for _, row in docs.iterrows():
        print("\n---")
        print("TITLE:", row.get("product_title"))
        print("TEXT:", row.get("chunk_text")[:200])
