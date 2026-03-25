import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from pathlib import Path
from datetime import datetime

# Directories
ORIGINAL_DIR = "RAG_knowledge/original"
INDEX_DIR = "RAG_knowledge/index"

# Single config
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512


def chunk_text_by_tokens(text: str, tokenizer, chunk_size: int, overlap: int = 50):
    """Split text into chunks of approximately chunk_size tokens with overlap."""
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start = end - overlap if overlap else end
    return chunks


def extract_header_info(text: str):
    """Extract markdown header hierarchy from text for chunk metadata."""
    lines = text.split("\n")
    current_h1 = ""
    current_h2 = ""
    current_h3 = ""

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("### "):
            current_h3 = stripped[4:]
        elif stripped.startswith("## "):
            current_h2 = stripped[3:]
            current_h3 = ""
        elif stripped.startswith("# "):
            current_h1 = stripped[2:]
            current_h2 = ""
            current_h3 = ""

    header_parts = [current_h1, current_h2, current_h3]
    header = " > ".join(p for p in header_parts if p)
    return header, current_h1, current_h2, current_h3


def chunk_text_with_headers(text: str, tokenizer, chunk_size: int, overlap: int = 50):
    """Split text into chunks and track which headers each chunk falls under."""
    lines = text.split("\n")
    current_h1 = ""
    current_h2 = ""
    current_h3 = ""

    # Build sections: accumulate text between headers
    sections = []
    current_text_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("### "):
            if current_text_lines:
                sections.append({
                    "text": "\n".join(current_text_lines),
                    "h1": current_h1, "h2": current_h2, "h3": current_h3
                })
                current_text_lines = []
            current_h3 = stripped[4:]
        elif stripped.startswith("## "):
            if current_text_lines:
                sections.append({
                    "text": "\n".join(current_text_lines),
                    "h1": current_h1, "h2": current_h2, "h3": current_h3
                })
                current_text_lines = []
            current_h2 = stripped[3:]
            current_h3 = ""
        elif stripped.startswith("# "):
            if current_text_lines:
                sections.append({
                    "text": "\n".join(current_text_lines),
                    "h1": current_h1, "h2": current_h2, "h3": current_h3
                })
                current_text_lines = []
            current_h1 = stripped[2:]
            current_h2 = ""
            current_h3 = ""
        else:
            if stripped:  # skip empty lines
                current_text_lines.append(line)

    # Don't forget the last section
    if current_text_lines:
        sections.append({
            "text": "\n".join(current_text_lines),
            "h1": current_h1, "h2": current_h2, "h3": current_h3
        })

    # Now chunk each section's text and carry header info
    chunks = []
    for section in sections:
        section_text = section["text"].strip()
        if not section_text:
            continue

        token_chunks = chunk_text_by_tokens(section_text, tokenizer, chunk_size, overlap)
        header_parts = [section["h1"], section["h2"], section["h3"]]
        header = " > ".join(p for p in header_parts if p)

        for chunk_text in token_chunks:
            chunks.append({
                "text": chunk_text,
                "header": header,
                "h1": section["h1"],
                "h2": section["h2"],
                "h3": section["h3"],
            })

    return chunks


def process_file(md_path: str, model: SentenceTransformer, tokenizer):
    """Process a single markdown file: chunk, embed, save index."""
    md_path = Path(md_path)
    file_name = md_path.stem  # e.g. "fast_speaking_rate"
    category = md_path.parent.name  # e.g. "bad_habits"
    relative_path = str(md_path.relative_to(ORIGINAL_DIR))

    out_dir = os.path.join(INDEX_DIR, file_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nProcessing: {relative_path}")

    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Chunk with header tracking
    chunks = chunk_text_with_headers(text, tokenizer, CHUNK_SIZE)

    if not chunks:
        print(f"  Warning: No chunks generated for {file_name}")
        return None

    # Generate embeddings
    chunk_texts = [c["text"] for c in chunks]
    embeddings = model.encode(chunk_texts, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))

    # Calculate stats
    total_words = sum(len(c["text"].split()) for c in chunks)

    # Save chunks metadata
    metadata = {
        "source_file": str(md_path),
        "file_name": file_name,
        "category": category,
        "relative_path": relative_path,
        "model": MODEL_NAME,
        "chunk_size_tokens": CHUNK_SIZE,
        "num_chunks": len(chunks),
        "embedding_dim": dimension,
        "stats": {
            "total_words": total_words,
            "indexed_date": datetime.now().isoformat(),
        },
        "chunks": chunks,
    }

    with open(os.path.join(out_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  {len(chunks)} chunks, {dimension}d embeddings, {total_words} words")
    return file_name, category


if __name__ == "__main__":
    print(f"Embedding model: {MODEL_NAME}")
    print(f"Chunk size: {CHUNK_SIZE} tokens")
    print(f"Source: {ORIGINAL_DIR}")
    print(f"Output: {INDEX_DIR}")

    # Load model and tokenizer once
    model = SentenceTransformer(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{MODEL_NAME}")

    # Find all markdown files recursively
    md_files = sorted(Path(ORIGINAL_DIR).rglob("*.md"))
    print(f"\nFound {len(md_files)} markdown files")

    # Process each file
    category_mapping = {}
    for md_file in md_files:
        result = process_file(str(md_file), model, tokenizer)
        if result:
            file_name, category = result
            category_mapping[file_name] = category

    # Save category mapping
    mapping_path = os.path.join(INDEX_DIR, "category_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(category_mapping, f, indent=2, ensure_ascii=False)

    print(f"\nCategory mapping saved: {category_mapping}")
    print(f"\nAll {len(md_files)} files indexed successfully!")
