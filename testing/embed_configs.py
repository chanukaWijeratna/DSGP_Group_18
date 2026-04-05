import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

DATA_DIR = "original_data"
OUTPUT_DIR = "embeddings"

CONFIGS = [
    ("all-MiniLM-L6-v2", 256),
    ("all-MiniLM-L6-v2", 512),
    ("all-MiniLM-L6-v2", 512),  # duplicate in table but with overlap variant - we'll skip exact dupes
    ("all-mpnet-base-v2", 256),
    ("all-mpnet-base-v2", 512),
    ("all-mpnet-base-v2", 512),
]

# Deduplicate
CONFIGS = list(set(CONFIGS))
CONFIGS.sort(key=lambda x: (x[0], x[1]))
print(f"Unique configs: {CONFIGS}")


def chunk_text_by_tokens(text: str, tokenizer, chunk_size: int, overlap: int = 0):
    """Split text into chunks of approximately chunk_size tokens."""
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


def process_config(model_name: str, chunk_size: int):
    folder_name = f"{model_name}_chunk{chunk_size}"
    out_dir = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Config: model={model_name}, chunk_size={chunk_size}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    # Load model and tokenizer
    model = SentenceTransformer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{model_name}")

    md_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".md")]
    md_files.sort()

    for md_file in md_files:
        filepath = os.path.join(DATA_DIR, md_file)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text_by_tokens(text, tokenizer, chunk_size)
        embeddings = model.encode(chunks)

        base_name = os.path.splitext(md_file)[0]

        # Save embeddings as .npy
        np.save(os.path.join(out_dir, f"{base_name}_embeddings.npy"), embeddings)

        # Save chunks as JSON for reference
        metadata = {
            "source_file": md_file,
            "model": model_name,
            "chunk_size_tokens": chunk_size,
            "num_chunks": len(chunks),
            "embedding_dim": embeddings.shape[1],
            "chunks": chunks,
        }
        with open(os.path.join(out_dir, f"{base_name}_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"  {md_file}: {len(chunks)} chunks, embedding shape {embeddings.shape}")

    print(f"Done: {folder_name}")


if __name__ == "__main__":
    for model_name, chunk_size in CONFIGS:
        process_config(model_name, chunk_size)
    print("\nAll configurations complete!")
