"""
Document loading and chunking for PaperMind.

Loads every PDF in DATASET_DIR, attaches paper-name metadata to each page,
then splits pages into overlapping chunks ready for embedding.
"""
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config


def load_pdfs(dataset_dir: Path = config.DATASET_DIR) -> List[Document]:
    """
    Load all PDFs from *dataset_dir*.

    Returns a flat list of LangChain Document objects, one per page.
    Each document carries metadata:
        - source      : original file path
        - page        : 0-based page index
        - paper_name  : PDF file stem (used as lookup key in PAPER_TITLES)
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    pdf_files = sorted(dataset_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {dataset_dir}")

    documents: List[Document] = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["paper_name"] = pdf_path.stem
        documents.extend(pages)
        print(f"  Loaded {len(pages):>3} pages  ←  {pdf_path.name}")

    print(f"\nTotal pages loaded: {len(documents)}")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks using a recursive character splitter.

    Splitting order: paragraph breaks → line breaks → spaces → characters.
    This preserves sentence and paragraph boundaries wherever possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Attach a human-readable title to every chunk for display in the UI
    for chunk in chunks:
        paper_name = chunk.metadata.get("paper_name", "unknown")
        chunk.metadata["paper_title"] = config.PAPER_TITLES.get(
            paper_name, paper_name.replace("_", " ").title()
        )

    print(f"Total chunks created: {len(chunks)}")
    return chunks
