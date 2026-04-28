"""
ChromaDB vector store management for PaperMind.

Provides helpers to:
  - initialise the HuggingFace embedding model (local, free)
  - create and persist a new vector store from document chunks
  - load an existing persisted vector store
  - check whether a store already exists
"""
import pickle
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import config

# Path where BM25 chunks are cached for hybrid retrieval
_CHUNKS_CACHE = config.VECTOR_STORE_DIR / "chunks.pkl"


# Embedding model

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a HuggingFace embedding model.

    BAAI/bge-large-en-v1.5 is used by default; it produces 1024-dim vectors
    and consistently ranks near the top of the MTEB leaderboard.
    normalize_embeddings=True ensures cosine similarity is equivalent to
    dot-product similarity.
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# Vector store lifecycle

def create_vector_store(chunks: List[Document]) -> Chroma:
    """
    Embed *chunks* and persist a new ChromaDB collection.

    Also caches the raw chunks to disk so the hybrid BM25 retriever can
    reload them without re-parsing the PDFs.
    """
    config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(config.VECTOR_STORE_DIR),
        collection_name=config.COLLECTION_NAME,
    )

    # Cache chunks for BM25 hybrid retrieval
    with open(_CHUNKS_CACHE, "wb") as fh:
        pickle.dump(chunks, fh)

    print(f"Vector store persisted at: {config.VECTOR_STORE_DIR}")
    return vector_store


def load_vector_store() -> Chroma:
    """Load an existing persisted vector store."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=str(config.VECTOR_STORE_DIR),
        embedding_function=embeddings,
        collection_name=config.COLLECTION_NAME,
    )


def load_chunks() -> Optional[List[Document]]:
    """Return cached chunks if available (used by hybrid retriever)."""
    if _CHUNKS_CACHE.exists():
        with open(_CHUNKS_CACHE, "rb") as fh:
            return pickle.load(fh)
    return None


def vector_store_exists() -> bool:
    """Return True if a persisted vector store is present."""
    return (
        config.VECTOR_STORE_DIR.exists()
        and any(config.VECTOR_STORE_DIR.iterdir())
    )
