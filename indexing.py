import os
import json
import faiss
import pickle
from datetime import datetime
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

# =============================================================================
# CONFIG
# =============================================================================

KNOWLEDGE_DIR = "RAG_knowledge/original"
OUTPUT_DIR = "RAG_knowledge/index"
CHUNK_SIZE = 150  # Reduced from 250 for better focused chunks
CHUNK_OVERLAP = 35  # Proportionally reduced
MIN_CHUNK_SIZE = 50  # Minimum words for a valid chunk

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_markdown(file_path: str) -> str:
    """Load markdown file content."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise


def extract_document_structure(text: str) -> List[Dict]:
    """
    Parse markdown and extract sections with their hierarchical headers.
    Returns list of sections with header context.
    """
    lines = text.split('\n')
    sections = []
    current_headers = {'h1': '', 'h2': '', 'h3': ''}
    current_content = []
    
    for line in lines:
        # Update header hierarchy
        if line.startswith('### '):
            current_headers['h3'] = line[4:].strip()
        elif line.startswith('## '):
            current_headers['h2'] = line[3:].strip()
            current_headers['h3'] = ''
        elif line.startswith('# '):
            current_headers['h1'] = line[2:].strip()
            current_headers['h2'] = ''
            current_headers['h3'] = ''
        else:
            current_content.append(line)
        
        # When we hit section boundary or end, save the section
        if (line.startswith('#') or line.strip() == '') and current_content:
            content_text = '\n'.join(current_content).strip()
            if content_text and len(content_text.split()) >= MIN_CHUNK_SIZE:
                # Build header path
                header_parts = [h for h in [current_headers['h1'], 
                                           current_headers['h2'], 
                                           current_headers['h3']] if h]
                header_path = ' > '.join(header_parts)
                
                sections.append({
                    'header': header_path,
                    'h1': current_headers['h1'],
                    'h2': current_headers['h2'],
                    'h3': current_headers['h3'],
                    'content': content_text
                })
            current_content = []
    
    # Handle remaining content
    if current_content:
        content_text = '\n'.join(current_content).strip()
        if content_text and len(content_text.split()) >= MIN_CHUNK_SIZE:
            header_parts = [h for h in [current_headers['h1'], 
                                       current_headers['h2'], 
                                       current_headers['h3']] if h]
            header_path = ' > '.join(header_parts)
            
            sections.append({
                'header': header_path,
                'h1': current_headers['h1'],
                'h2': current_headers['h2'],
                'h3': current_headers['h3'],
                'content': content_text
            })
    
    return sections


def chunk_section(section: Dict, chunk_size: int, overlap: int) -> List[Dict]:
    """
    Split a section into overlapping chunks while preserving header context.
    """
    content = section['content']
    words = content.split()
    chunks = []
    
    if len(words) <= chunk_size:
        # Section is small enough, return as single chunk
        return [{
            'text': content,
            'header': section['header'],
            'h1': section['h1'],
            'h2': section['h2'],
            'h3': section['h3']
        }]
    
    # Split into overlapping chunks
    start = 0
    chunk_num = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)
        
        # Add header context to chunk for better retrieval
        if section['header']:
            chunk_with_context = f"[{section['header']}]\n\n{chunk_text}"
        else:
            chunk_with_context = chunk_text
        
        chunks.append({
            'text': chunk_with_context,
            'original_text': chunk_text,  # Without header prefix
            'header': section['header'],
            'h1': section['h1'],
            'h2': section['h2'],
            'h3': section['h3'],
            'chunk_num': chunk_num,
            'is_partial': True
        })
        
        chunk_num += 1
        start += chunk_size - overlap
    
    return chunks


def chunk_text_structured(text: str, chunk_size: int, overlap: int) -> List[Dict]:
    """
    Main chunking function that respects document structure.
    """
    sections = extract_document_structure(text)
    all_chunks = []
    
    for section in sections:
        section_chunks = chunk_section(section, chunk_size, overlap)
        all_chunks.extend(section_chunks)
    
    return all_chunks


# =============================================================================
# INDEXING LOGIC
# =============================================================================

def index_markdown_file(md_path: str) -> bool:
    """
    Index a single markdown file with FAISS and save metadata.
    Returns True on success, False on failure.
    """
    try:
        file_name = os.path.splitext(os.path.basename(md_path))[0]
        print(f"Indexing: {file_name}")
        
        # Load and parse document
        text = load_markdown(md_path)
        chunks = chunk_text_structured(text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        if not chunks:
            print(f"  Warning: No chunks generated for {file_name}")
            return False
        
        print(f"  Generated {len(chunks)} chunks")
        
        # Extract text for embedding
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        print(f"  Generating embeddings...")
        embeddings = embedding_model.encode(
            chunk_texts, 
            batch_size=32, 
            show_progress_bar=False
        )
        
        # Create FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        
        # Create output directory
        file_output_dir = os.path.join(OUTPUT_DIR, file_name)
        os.makedirs(file_output_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(index, os.path.join(file_output_dir, "index.faiss"))
        
        # Prepare comprehensive metadata
        metadata = {
            "source_file": md_path,
            "file_name": file_name,
            "num_chunks": len(chunks),
            "chunks": chunks,
            "config": {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "min_chunk_size": MIN_CHUNK_SIZE,
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dimension": dim
            },
            "stats": {
                "total_words": len(text.split()),
                "total_chars": len(text),
                "file_size_bytes": os.path.getsize(md_path),
                "indexed_date": datetime.now().isoformat()
            }
        }
        
        # Save metadata
        with open(os.path.join(file_output_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved index to {file_output_dir}\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Error indexing {md_path}: {e}\n")
        return False


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("RAG Knowledge Base Indexer")
    print("="*80)
    print(f"Knowledge Directory: {KNOWLEDGE_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Chunk Size: {CHUNK_SIZE} words")
    print(f"Chunk Overlap: {CHUNK_OVERLAP} words")
    print(f"Embedding Model: all-MiniLM-L6-v2")
    print("="*80 + "\n")
    
    # Find all markdown files
    md_files = []
    for root, _, files in os.walk(KNOWLEDGE_DIR):
        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                md_files.append(full_path)
    
    if not md_files:
        print(f"No markdown files found in {KNOWLEDGE_DIR}")
        return
    
    print(f"Found {len(md_files)} markdown file(s)\n")
    
    # Index all files
    success_count = 0
    fail_count = 0
    
    for md_path in md_files:
        if index_markdown_file(md_path):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print("="*80)
    print("INDEXING COMPLETE")
    print("="*80)
    print(f"Successfully indexed: {success_count} file(s)")
    print(f"Failed: {fail_count} file(s)")
    print(f"Output location: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()