"""
ingest.py
---------
Reads all PDFs from a knowledge-base folder (organized into subfolders like
returns/, delivery/, refunds/, sellers/, payments/, customer_support/),
splits their text into overlapping chunks, embeds the chunks with a
sentence-transformers model, and saves everything into a FAISS index plus
a metadata file so you can do retrieval later.

Usage (from a Colab cell or terminal):
    python ingest.py --input_dir /content/daraz_knowledge_base --output_dir /content/faiss_index

Outputs (in --output_dir):
    index.faiss       -> the FAISS vector index
    metadata.pkl      -> list of dicts: {"text": chunk, "source": path, "category": subfolder, "chunk_id": int}
"""

import os
import re
import pickle
import argparse
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


# ---------------------------------------------------------------------------
# 1. PDF text extraction
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a single PDF file, page by page."""
    text_parts = []
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    except Exception as e:
        print(f"  [warn] failed to read {pdf_path}: {e}")
    return "\n".join(text_parts)


def clean_text(text: str) -> str:
    """Light cleanup: collapse whitespace, strip weird control chars."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150):
    """
    Simple sliding-window word-based chunker.
    chunk_size / chunk_overlap are measured in words (a reasonable proxy
    for tokens without pulling in a tokenizer dependency).
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = max(chunk_size - chunk_overlap, 1)
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start += step
    return chunks


# ---------------------------------------------------------------------------
# 3. Walk the folder, extract + chunk everything
# ---------------------------------------------------------------------------
def collect_chunks(input_dir: str, chunk_size: int, chunk_overlap: int):
    input_dir = Path(input_dir)
    all_chunks = []  # list of dicts: text, source, category, chunk_id

    pdf_paths = sorted(input_dir.rglob("*.pdf"))
    if not pdf_paths:
        print(f"[warn] no PDFs found under {input_dir}")

    for pdf_path in pdf_paths:
        # category = immediate subfolder name relative to input_dir
        try:
            relative = pdf_path.relative_to(input_dir)
            category = relative.parts[0] if len(relative.parts) > 1 else "uncategorized"
        except ValueError:
            category = "uncategorized"

        print(f"Reading: {pdf_path}  (category: {category})")
        raw_text = extract_text_from_pdf(str(pdf_path))
        raw_text = clean_text(raw_text)

        if not raw_text:
            print(f"  [warn] no extractable text in {pdf_path}")
            continue

        chunks = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": str(pdf_path),
                "category": category,
                "chunk_id": i,
            })

        print(f"  -> {len(chunks)} chunks")

    return all_chunks


# ---------------------------------------------------------------------------
# 4. Embed + build FAISS index
# ---------------------------------------------------------------------------
def build_faiss_index(chunks, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so we can use inner-product = cosine similarity
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
    index.add(embeddings)

    return index, embeddings


# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
def save_index(index, chunks, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(output_dir / "index.faiss"))
    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved FAISS index -> {output_dir / 'index.faiss'}")
    print(f"Saved metadata    -> {output_dir / 'metadata.pkl'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into a FAISS index.")
    parser.add_argument("--input_dir", type=str, required=True,
                         help="Path to daraz_knowledge_base folder (with subfolders).")
    parser.add_argument("--output_dir", type=str, default="./faiss_index",
                         help="Where to save index.faiss and metadata.pkl")
    parser.add_argument("--chunk_size", type=int, default=800,
                         help="Chunk size in words")
    parser.add_argument("--chunk_overlap", type=int, default=150,
                         help="Overlap between chunks in words")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2",
                         help="sentence-transformers model to use for embeddings")
    args = parser.parse_args()

    chunks = collect_chunks(args.input_dir, args.chunk_size, args.chunk_overlap)
    if not chunks:
        print("No chunks were created — check that your input_dir has readable PDFs.")
        return

    index, _ = build_faiss_index(chunks, model_name=args.model_name)
    save_index(index, chunks, args.output_dir)

    print(f"\nDone. Indexed {len(chunks)} chunks from {args.input_dir}.")


if __name__ == "__main__":
    main()
