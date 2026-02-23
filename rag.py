import faiss
import json
import requests
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import List, Dict, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ENDPOINT = "https://openrouter.ai/api/v1"
API_KEY = "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"
LLM_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"

TOP_K = 6
MIN_SIMILARITY_THRESHOLD = 0.3  # Filter out irrelevant chunks

# Initialize models
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

client = OpenAI(
    base_url=ENDPOINT,
    api_key=API_KEY,
)

# Response format template
RESPONSE_FORMAT = """
### What This Means

{2–4 sentences explanation of how problem words affect clarity, credibility, and listener perception.}

---

### Why This May Be Happening

{Short explanation based on context: reasons why problem is arising}

---

### Actionable Improvement Steps

- {Clear, practical action with specific technique}
- {Clear, practical action with specific technique}
- {Clear, practical action with specific technique}

---

### Quick Practice Exercise

{1 simple exercise the user can apply immediately, with concrete example.}

---

### Quick Note

{1–2 positive, confidence-supporting sentences that acknowledge the challenge while encouraging progress.}
"""


# =============================================================================
# INDEX LOADING
# =============================================================================

def load_index(index_dir: str) -> Tuple[faiss.Index, List[Dict]]:
    """
    Load FAISS index and chunk metadata.
    
    Args:
        index_dir: Path to the index directory
        
    Returns:
        Tuple of (faiss_index, chunks_list)
    """
    try:
        index = faiss.read_index(f"{index_dir}/index.faiss")
        
        with open(f"{index_dir}/chunks.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        chunks = metadata["chunks"]
        print(f"✓ Loaded index with {len(chunks)} chunks")
        return index, chunks
    
    except FileNotFoundError as e:
        print(f"✗ Error: Index files not found in {index_dir}")
        print(f"  Make sure you've run improved_rag_indexer.py first")
        raise
    except Exception as e:
        print(f"✗ Error loading index: {e}")
        raise


# =============================================================================
# RETRIEVAL
# =============================================================================

def retrieve_chunks(
    query: str, 
    index: faiss.Index, 
    chunks: List[Dict], 
    top_k: int = TOP_K
) -> List[Dict]:
    """
    Retrieve most relevant chunks for the query.
    
    Args:
        query: User's question or problem description
        index: FAISS index
        chunks: List of chunk dictionaries
        top_k: Number of chunks to retrieve
        
    Returns:
        List of retrieved chunks with similarity scores
    """
    # Encode query
    query_embedding = embedding_model.encode([query])
    
    # Search
    distances, indices = index.search(query_embedding, top_k)
    
    # Format results with similarity scores
    retrieved = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            # Convert L2 distance to similarity score
            similarity = 1 / (1 + distance)
            
            # Filter by threshold
            if similarity >= MIN_SIMILARITY_THRESHOLD:
                chunk = chunks[idx].copy()
                chunk['similarity'] = float(similarity)
                chunk['distance'] = float(distance)
                retrieved.append(chunk)
    
    print(f"✓ Retrieved {len(retrieved)} relevant chunks")
    return retrieved


def format_chunks_for_context(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a readable context string.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, chunk in enumerate(chunks, 1):
        # Get the original text without header prefix
        text = chunk.get('original_text', chunk.get('text', ''))
        header = chunk.get('header', 'General Content')
        similarity = chunk.get('similarity', 0)
        
        context_parts.append(
            f" [{header}] \n{text}"
        )
    
    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# LLM INTERACTION
# =============================================================================

def call_llm(context_chunks: List[Dict], user_query: str, verbose: bool = False) -> str:
    """
    Call LLM with retrieved context to generate response.
    
    Args:
        context_chunks: Retrieved chunks with metadata
        user_query: User's original query
        verbose: Whether to print the full prompt
        
    Returns:
        LLM response text
    """
    # Format context
    context = format_chunks_for_context(context_chunks)
    
    # Build prompt
    prompt = f"""You are a supportive and knowledgeable speech feedback assistant specializing in verbal fluency.

**CRITICAL INSTRUCTIONS:**
- Use ONLY the information explicitly provided in the CONTEXT section.
- Do NOT introduce concepts, causes, techniques, or examples that are not directly supported by the context.
- Do NOT mention medical diagnoses, disorders, or treatments unless explicitly stated in the context.
- If the context does not provide enough information for a section, state that briefly instead of guessing.
- Keep the response under 300 words total.
- Maintain a supportive, practical, non-judgmental tone.
- Frame all guidance as non-diagnostic, general speaking support.

**OUTPUT FORMAT (FOLLOW EXACTLY):**
{RESPONSE_FORMAT}

**CONTEXT FROM KNOWLEDGE BASE:**
{context}

**USER REQUEST:**
{user_query}

**YOUR RESPONSE:**
"""

    print("\n" + "="*80)
    print("FULL PROMPT SENT TO LLM")
    print("="*80)
    print(prompt)
    print("="*80 + "\n")
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
            max_tokens=1000,
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"✗ Error calling LLM: {e}")
        raise


# =============================================================================
# MAIN RAG PIPELINE
# =============================================================================

def run_rag(
    index_dir: str, 
    user_query: str, 
    top_k: int = TOP_K,
    verbose: bool = False
) -> Dict:
    """
    Run the complete RAG pipeline.
    
    Args:
        index_dir: Path to the index directory
        user_query: User's question or problem
        top_k: Number of chunks to retrieve
        verbose: Whether to print debug information
        
    Returns:
        Dictionary with response and metadata
    """
    print(f"\n{'='*80}")
    print("RAG Speech Feedback System")
    print(f"{'='*80}\n")
    
    # Load index
    print("Loading index...")
    index, chunks = load_index(index_dir)
    
    # Retrieve relevant chunks
    print(f"\nSearching for relevant content...")
    retrieved_chunks = retrieve_chunks(user_query, index, chunks, top_k)
    
    if not retrieved_chunks:
        return {
            "response": "I couldn't find relevant information in the knowledge base to answer your question",
            "retrieved_chunks": [],
            "query": user_query
        }
    
    # Display retrieved sources
    print("\nRetrieved sources:")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"  {i}. {chunk.get('header', 'Unknown')} (similarity: {chunk.get('similarity', 0):.2%})")
    
    # Generate response
    print(f"\nGenerating response using {LLM_MODEL}...")
    response = call_llm(retrieved_chunks, user_query, verbose=verbose)
    
    print("\n✓ Response generated successfully\n")
    
    return {
        "response": response,
        "retrieved_chunks": retrieved_chunks,
        "query": user_query,
        "num_chunks": len(retrieved_chunks)
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    INDEX_PATH = "RAG_knowledge/index/nervouseness"
    problem = "The user shows nervouseness when speaking"

    # INDEX_PATH = "RAG_knowledge/index/filler_words"
    # problem = "The user uses too many filler words when speaking"

    # INDEX_PATH = "RAG_knowledge/index/stuttering"
    # problem = "The user stutters a lot when speaking"

    request = "Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise"
    
    query = f"{problem}. {request}"

    result = run_rag(INDEX_PATH, query, verbose=False)['response']

    result.replace("}", "")
    result.replace("{", "")

    print("="*80)
    print("RESPONSE")
    print("="*80 + "\n")
    print(result.strip("{}"))