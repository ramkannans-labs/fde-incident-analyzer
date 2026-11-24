"""Retrieval layer: BM25 keyword search + optional semantic embeddings."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Optional

from ..schemas import LogChunk, RetrievedChunk


class BM25Index:
    """Simple BM25 implementation for keyword-based retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[List[str]] = []
        self._doc_ids: List[str] = []
        self._doc_lengths: List[int] = []
        self._avg_dl: float = 0.0
        self._idf: dict[str, float] = {}
        self._total_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words - split on hyphens and dots too."""
        # Replace hyphens and dots with spaces, then find words
        text = text.replace('-', ' ').replace('.', ' ')
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    def add(self, chunk: LogChunk) -> None:
        """Add a chunk to the index."""
        tokens = self._tokenize(chunk.text)
        self._docs.append(tokens)
        self._doc_ids.append(chunk.chunk_id)
        self._doc_lengths.append(len(tokens))
        self._total_docs += 1

    def build(self) -> None:
        """Build the index (must call after adding all docs)."""
        N = len(self._docs)
        if N == 0:
            return

        # Compute average document length
        self._avg_dl = sum(self._doc_lengths) / N

        # Compute IDF for each term
        df: dict[str, int] = {}
        for doc in self._docs:
            seen = set()
            for term in doc:
                if term not in seen:
                    df[term] = df.get(term, 0) + 1
                    seen.add(term)

        self._idf = {}
        for term, count in df.items():
            idf = math.log(1 + (N - count + 0.5) / (count + 0.5))
            self._idf[term] = idf

    def search(self, query: str, top_k: int = 10) -> List[tuple[str, float]]:
        """Search and return (chunk_id, score) pairs."""
        if not self._docs:
            return []

        query_terms = self._tokenize(query)
        scores: dict[str, float] = {}

        for i, doc in enumerate(self._docs):
            doc_len = self._doc_lengths[i]
            tf = Counter(doc)
            score = 0.0

            for term in query_terms:
                if term not in tf:
                    continue
                term_freq = tf[term]
                idf = self._idf.get(term, 0.0)
                # BM25 scoring
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / self._avg_dl)
                score += idf * numerator / denominator

            if score > 0:
                scores[self._doc_ids[i]] = score

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked


class HybridRetriever:
    """Hybrid retrieval: BM25 keyword + optional semantic embeddings."""

    def __init__(self, use_semantic: bool = False, ollama_base_url: str = "http://localhost:11434"):
        self.bm25 = BM25Index()
        self.chunks: dict[str, LogChunk] = {}  # chunk_id -> chunk
        self.use_semantic = use_semantic
        self.ollama_base_url = ollama_base_url
        self._semantic_index = None  # Optional: Qdrant client

    def index(self, chunks: List[LogChunk]) -> None:
        """Index a list of chunks."""
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            self.bm25.add(chunk)
        self.bm25.build()

    def search(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        """Search using BM25 + optional semantic reranking."""
        # BM25 retrieval
        bm25_results = self.bm25.search(query, top_k=top_k * 2)

        if not self.use_semantic or not bm25_results:
            return [
                RetrievedChunk(
                    chunk=self.chunks[cid],
                    score=min(score / (bm25_results[0][1] if bm25_results else 1.0), 1.0),
                    match_type="keyword",
                )
                for cid, score in bm25_results[:top_k]
            ]

        # Semantic reranking (optional)
        try:
            semantic_scores = self._semantic_rerank(query, bm25_results[:top_k])
        except Exception:
            semantic_scores = {}

        results = []
        for cid, bm25_score in bm25_results[:top_k]:
            sem_score = semantic_scores.get(cid, 0.0)
            # Combine scores (60% BM25, 40% semantic)
            combined = 0.6 * (bm25_score / (bm25_results[0][1] if bm25_results else 1.0)) + 0.4 * sem_score
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[cid],
                    score=min(combined, 1.0),
                    match_type="hybrid",
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _semantic_rerank(self, query: str, candidates: List[tuple[str, float]]) -> dict[str, float]:
        """Use Ollama embeddings for semantic similarity."""
        # This is a stub - real implementation uses Ollama API
        # For now, return empty dict (falls back to BM25 only)
        return {}


def build_retriever(chunks: List[LogChunk], use_semantic: bool = False) -> HybridRetriever:
    """Build a retriever from chunks."""
    retriever = HybridRetriever(use_semantic=use_semantic)
    retriever.index(chunks)
    return retriever