"""
Retrieval strategies for PaperMind.

Three strategies are available and selectable via config.RETRIEVAL_STRATEGY:

  1. similarity  – plain cosine similarity (fast baseline)
  2. mmr         – Maximal Marginal Relevance
                   Fetches fetch_k candidates, then greedily picks k chunks
                   that are relevant to the query AND diverse relative to
                   each other.  Good default for most Q&A workloads.
  3. hybrid      – Reciprocal Rank Fusion of BM25 (keyword) + semantic
                   Combines sparse (exact-match) and dense (semantic) signals.
                   Best for short, keyword-heavy queries.
"""
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
# from langchain_core.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import config


# Individual retrievers

def _similarity_retriever(vector_store: Chroma, k: int) -> BaseRetriever:
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def _mmr_retriever(vector_store: Chroma, k: int) -> BaseRetriever:
    """
    MMR retriever.

    fetch_k = k * 3  — oversample candidates before diversity filtering.
    lambda_mult       — 1.0 = full relevance, 0.0 = full diversity.
    """
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": k * 3,
            "lambda_mult": config.MMR_LAMBDA,
        },
    )


def _hybrid_retriever(
    vector_store: Chroma,
    chunks: List[Document],
    k: int,
) -> BaseRetriever:
    """
    Hybrid BM25 + semantic retriever using Reciprocal Rank Fusion.

    BM25 captures exact keyword matches; semantic search captures
    conceptual similarity.  Their ranked lists are fused with RRF so
    chunks that rank highly in *both* lists bubble to the top.
    """
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k

    semantic = _similarity_retriever(vector_store, k)

    return EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=config.HYBRID_WEIGHTS,
    )


# Public factory

def get_retriever(
    vector_store: Chroma,
    strategy: str = config.RETRIEVAL_STRATEGY,
    chunks: Optional[List[Document]] = None,
    k: int = config.TOP_K,
) -> BaseRetriever:
    """
    Return the appropriate retriever for *strategy*.

    Args:
        vector_store: A loaded ChromaDB Chroma instance.
        strategy:     One of "similarity", "mmr", or "hybrid".
        chunks:       Required when strategy=="hybrid" (BM25 needs raw text).
        k:            Number of chunks to return per query.
    """
    strategy = strategy.lower()

    if strategy == "mmr":
        return _mmr_retriever(vector_store, k)

    if strategy == "hybrid":
        if chunks is None:
            # Fall back gracefully if chunks are unavailable
            print(
                "Warning: hybrid retrieval requires cached chunks. "
                "Falling back to MMR."
            )
            return _mmr_retriever(vector_store, k)
        return _hybrid_retriever(vector_store, chunks, k)

    # Default: plain cosine similarity
    return _similarity_retriever(vector_store, k)
