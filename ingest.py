"""
PaperMind – Document Ingestion Script
======================================
Run this to:
  1. Load all PDFs from Dataset/ either ovce or when documents change
  2. Split them into overlapping chunks
  3. Embed each chunk with BAAI/bge-large-en-v1.5 (local model)
  4. save the embeddings in a ChromaDB vector store
    """
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import config
from src.document_processor import load_pdfs, chunk_documents
from src.vector_store import create_vector_store, vector_store_exists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index research papers into PaperMind's vector store."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index even if a vector store already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 55)
    print("  PaperMind – Document Ingestion")
    print("=" * 55)

    if vector_store_exists() and not args.force:
        print("\nVector store already exists.")
        print("Run with --force to re-index.\n")
        sys.exit(0)

    # Validate dataset
    if not config.DATASET_DIR.exists():
        print(f"\nERROR: Dataset directory not found: {config.DATASET_DIR}")
        sys.exit(1)

    pdf_files = sorted(config.DATASET_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"\nERROR: No PDF files found in {config.DATASET_DIR}")
        sys.exit(1)

    print(f"\nFound {len(pdf_files)} PDF file(s) in {config.DATASET_DIR.name}/:")
    for f in pdf_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        title = config.PAPER_TITLES.get(f.stem, f.stem)
        print(f"  [{size_mb:5.1f} MB]  {title}")

    # Step 1 – Load
    print("\n[1/3] Loading PDFs …")
    documents = load_pdfs()

    # Step 2 – Chunk
    print(f"\n[2/3] Chunking  (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}) …")
    chunks = chunk_documents(documents)

    # Step 3 – Embed & persist
    print(f"\n[3/3] Embedding with '{config.EMBEDDING_MODEL}' and persisting …")
    print("      (First run downloads the model — this may take a minute.)\n")
    create_vector_store(chunks)

    # Summary
    print("\n" + "=" * 55)
    print("  Ingestion complete!")
    print("=" * 55)
    print(f"  Papers indexed : {len(pdf_files)}")
    print(f"  Total chunks   : {len(chunks)}")
    print(f"  Vector store   : {config.VECTOR_STORE_DIR}")
    print("\nYou can now start the chatbot:")
    print("  streamlit run app.py\n")


if __name__ == "__main__":
    main()
