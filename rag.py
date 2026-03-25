import faiss
import json
from collections import OrderedDict
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import List, Dict, Tuple, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

import os

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ENDPOINT = "https://openrouter.ai/api/v1"
LLM_MODEL = "anthropic/claude-haiku-4.5"
INDEX_BASE_DIR = "RAG_knowledge/index"

def load_api_key():
    """Load API key from config.json file."""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"\n✗ config.json not found at {config_path}\n"
            f"Please create a config.json file with your API key:\n"
            f'{{\n    "api_key": "your-api-key-here"\n}}'
        )

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            return config.get("api_key")
    except json.JSONDecodeError:
        raise ValueError("config.json is not valid JSON")

API_KEY = load_api_key()

TOP_K = 3
MIN_SIMILARITY_THRESHOLD = 0.3  # Filter out irrelevant chunks

# =============================================================================
# PROBLEM-TO-INDEX MAPPING
# =============================================================================

# Maps keywords from emotion API's detected_problems to RAG index directories
EMOTION_PROBLEM_MAP = {
    "Stress/Anxiety Tone": "nervouseness",
    "Negative Emotional Dominance": "negative_emotion",
    "Flat/Low Expressiveness": "monotone_voice",
    "Emotion Inconsistency": "emotion_inconsistency",
}

# Maps disorder API flagged fields to RAG index directories
DISORDER_PROBLEM_MAP = {
    "stuttering": "stuttering",
    "slurring": "dysathria",
}

# Maps bad-habit detection flagged fields to RAG index directories
BAD_HABIT_PROBLEM_MAP = {
    "filler_detection": "filler_words",
    "pace_analysis": "fast_speaking_rate",
}

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

# Critical Instruction template
CRITICAL_INSTRUCTION = """
- Use ONLY the information explicitly provided in the CONTEXT section.
- Do NOT introduce concepts, causes, techniques, or examples that are not directly supported by the context.
- Do NOT mention medical diagnoses, disorders, or treatments.
- If the context does not provide enough information for a section, provide general practical advice instead of stating that information is unavailable. Never tell the user that something is missing from the context.
- Keep the response under 300 words total.
- Maintain a supportive, practical, non-judgmental tone.
- Frame all guidance as non-diagnostic, general speaking support.
- Provide response as if you are directly responding to user, not reading off a script.
"""
def get_emotion_instruction(scenario: str = "") -> str:
    scenario_line = (
        "- Assess whether the dominant emotions are appropriate to the SCENARIO provided.\n"
        "- Do not try to find the scenario in the CONTEXT, use the CONTEXT for generalized advice only\n"
    ) if scenario.strip() else ""
    return f"""
{scenario_line}- Use ONLY the information explicitly provided in the CONTEXT section.
- Do NOT introduce concepts, causes, techniques, or examples that are not directly supported by the context.
- Do NOT mention medical diagnoses, disorders, or treatments.
- If the context does not provide enough information for a section, provide general practical advice instead of stating that information is unavailable. Never tell the user that something is missing from the context.
- Keep the response under 300 words total.
- Maintain a supportive, practical, non-judgmental tone.
- Provide response as if you are directly responding to user, not reading off a script.
"""

EMOTION_PROBLEMS = {"nervouseness", "negative_emotion", "emotion_inconsistency", "monotone_voice"}
DISORDER_PROBLEMS = {"stuttering", "dysathria"}
BAD_HABIT_PROBLEMS = {"filler_words", "fast_speaking_rate"}


# =============================================================================
# INDEX LOADING AND DISCOVERY
# =============================================================================

def load_category_mapping() -> Dict[str, str]:
    """
    Load the category mapping from the index directory.

    Returns:
        Dictionary mapping file names to categories
    """
    mapping_path = os.path.join(INDEX_BASE_DIR, "category_mapping.json")

    try:
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass

    return {}


def list_available_indices() -> Dict[str, List[str]]:
    """
    Discover all available indices and group them by category.

    Returns:
        Dictionary with categories as keys and list of index names as values
    """
    indices_by_category = {}

    if not os.path.exists(INDEX_BASE_DIR):
        return indices_by_category

    category_mapping = load_category_mapping()

    for index_name in os.listdir(INDEX_BASE_DIR):
        index_path = os.path.join(INDEX_BASE_DIR, index_name)

        # Skip non-directories and special files
        if not os.path.isdir(index_path) or index_name.startswith("."):
            continue

        # Check if it has required files
        if not (os.path.exists(os.path.join(index_path, "index.faiss")) and
                os.path.exists(os.path.join(index_path, "chunks.json"))):
            continue

        # Get category from mapping or metadata
        category = category_mapping.get(index_name, "unknown")

        if category not in indices_by_category:
            indices_by_category[category] = []

        indices_by_category[category].append(index_name)

    return indices_by_category


def get_index_info(index_name: str) -> Optional[Dict]:
    """
    Get metadata information for a specific index.

    Args:
        index_name: Name of the index

    Returns:
        Dictionary with index metadata or None if not found
    """
    chunks_file = os.path.join(INDEX_BASE_DIR, index_name, "chunks.json")

    try:
        if os.path.exists(chunks_file):
            with open(chunks_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                return {
                    "file_name": metadata.get("file_name"),
                    "category": metadata.get("category", "unknown"),
                    "num_chunks": metadata.get("num_chunks", 0),
                    "total_words": metadata.get("stats", {}).get("total_words", 0),
                    "indexed_date": metadata.get("stats", {}).get("indexed_date")
                }
    except Exception:
        pass

    return None


def load_index(index_dir: str) -> Tuple[faiss.Index, List[Dict], Dict]:
    """
    Load FAISS index and chunk metadata with category information.

    Args:
        index_dir: Path to the index directory

    Returns:
        Tuple of (faiss_index, chunks_list, metadata)
    """
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))

    with open(os.path.join(index_dir, "chunks.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    chunks = metadata["chunks"]

    return index, chunks, metadata



# =============================================================================
# RETRIEVAL
# =============================================================================

def retrieve_chunks(
    query: str,
    index: faiss.Index,
    chunks: List[Dict],
    metadata: Dict,
    top_k: int = TOP_K
) -> List[Dict]:
    """
    Retrieve most relevant chunks for the query with section-diverse selection.

    Fetches extra candidates from FAISS, then picks the best chunk from each
    unique h2 section (e.g. "Introduction", "Solutions", "Possible Reasons")
    before filling remaining slots by similarity. This ensures the LLM receives
    context covering different aspects of the topic rather than redundant chunks
    from the same section.

    Args:
        query: User's question or problem description
        index: FAISS index
        chunks: List of chunk dictionaries
        metadata: Index metadata including category information
        top_k: Number of chunks to retrieve

    Returns:
        List of retrieved chunks with similarity scores and category
    """
    # Fetch extra candidates so we have enough to diversify across sections
    candidate_k = min(len(chunks), max(top_k * 3, 10))
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, candidate_k)

    # Build scored candidate list filtered by threshold
    candidates = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            similarity = 1 / (1 + distance)
            if similarity >= MIN_SIMILARITY_THRESHOLD:
                chunk = chunks[idx].copy()
                chunk['similarity'] = float(similarity)
                chunk['distance'] = float(distance)
                chunk['category'] = metadata.get('category', 'unknown')
                chunk['source_file'] = metadata.get('file_name', 'unknown')
                candidates.append(chunk)

    # --- Section-diverse selection ---
    # Group candidates by h2 section header, preserving similarity order
    section_buckets: Dict[str, List[Dict]] = OrderedDict()
    for c in candidates:
        section = c.get('h2') or c.get('header', 'General')
        section_buckets.setdefault(section, []).append(c)

    retrieved = []

    # Round 1: pick the best (highest-similarity) chunk from each unique section
    for section, bucket in section_buckets.items():
        if len(retrieved) >= top_k:
            break
        retrieved.append(bucket.pop(0))  # already sorted by similarity

    # Round 2: fill remaining slots with next-best candidates across all sections
    if len(retrieved) < top_k:
        remaining = [c for bucket in section_buckets.values() for c in bucket]
        remaining.sort(key=lambda c: c['similarity'], reverse=True)
        for c in remaining:
            if len(retrieved) >= top_k:
                break
            retrieved.append(c)

    # Sort final selection by similarity for consistent context ordering
    retrieved.sort(key=lambda c: c['similarity'], reverse=True)

    return retrieved


def format_chunks_for_context(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a readable context string with source attribution.

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
        category = chunk.get('category', 'unknown')
        source_file = chunk.get('source_file', 'unknown')

        source_info = f"[{category} > {source_file}]"

        context_parts.append(
            f"{source_info}\n{header}\n\n{text}"
        )

    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# LLM INTERACTION
# =============================================================================

def call_llm(context_chunks: List[Dict], user_query: str, critical_instructions: str = "") -> Tuple[str, str]:
    """
    Call LLM with retrieved context to generate response.

    Args:
        context_chunks: Retrieved chunks with metadata
        user_query: User's original query
        critical_instructions: Problem-specific critical instructions for the LLM

    Returns:
        Tuple of (LLM response text, finish reason)
    """
    context = format_chunks_for_context(context_chunks)

    # Build prompt
    prompt = f"""You are a supportive and knowledgeable speech feedback assistant specializing in verbal fluency.

**CRITICAL INSTRUCTIONS:**
{critical_instructions}

**OUTPUT FORMAT (FOLLOW EXACTLY):**
{RESPONSE_FORMAT}

**CONTEXT FROM KNOWLEDGE BASE:**
{context}

**USER REQUEST:**
{user_query}

**YOUR RESPONSE:**
"""

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

    content = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    return content, finish_reason


# =============================================================================
# MAIN RAG PIPELINE
# =============================================================================

def run_rag(
    index_dir: str,
    user_query: str,
    top_k: int = TOP_K,
    critical_instructions: str = "",
) -> Dict:
    """
    Run the complete RAG pipeline with category-aware retrieval.

    Args:
        index_dir: Path to the index directory
        user_query: User's question or problem
        top_k: Number of chunks to retrieve
        critical_instructions: Problem-specific critical instructions for the LLM

    Returns:
        Dictionary with response and metadata
    """
    index, chunks, metadata = load_index(index_dir)
    retrieved_chunks = retrieve_chunks(user_query, index, chunks, metadata, top_k)

    if not retrieved_chunks:
        return {
            "response": "I couldn't find relevant information in the knowledge base to answer your question",
            "retrieved_chunks": [],
            "query": user_query,
            "metadata": metadata
        }

    response, finish_reason = call_llm(retrieved_chunks, user_query, critical_instructions=critical_instructions)

    return {
        "response": response,
        "finish_reason": finish_reason,
        "retrieved_chunks": retrieved_chunks,
        "query": user_query,
        "num_chunks": len(retrieved_chunks),
        "metadata": metadata
    }

# =============================================================================
# INTEGRATED FEEDBACK FROM MODEL OUTPUTS
# =============================================================================

def _identify_problems(emotion_result: Optional[Dict], disorder_result: Optional[Dict], bad_habit_result: Optional[Dict] = None) -> List[str]:
    """
    Parse the JSON outputs from all models and return a list of RAG index names
    that should be queried for feedback.
    """
    problems = []

    # --- Emotion API problems ---
    if emotion_result:
        detected = emotion_result.get("detected_problems", [])
        for problem_text in detected:
            # Skip entries that indicate no problem
            if "No significant emotional problem" in problem_text:
                continue
            for keyword, index_name in EMOTION_PROBLEM_MAP.items():
                if keyword in problem_text and index_name not in problems:
                    problems.append(index_name)

    # --- Disorder API problems ---
    if disorder_result:
        stutter_data = disorder_result.get("stuttering", {})
        slur_data = disorder_result.get("slurring", {})

        if stutter_data.get("flagged", False):
            problems.append("stuttering")
        if slur_data.get("flagged", False):
            problems.append("dysathria")

    # --- Bad Habit API problems ---
    if bad_habit_result:
        for field, index_name in BAD_HABIT_PROBLEM_MAP.items():
            if bad_habit_result.get(field, {}).get("flagged", False):
                problems.append(index_name)

    return problems


def _build_query(
    problem_index: str,
    scenario: str = "",
    disorder_result: Optional[Dict] = None,
    emotion_result: Optional[Dict] = None,
    bad_habit_result: Optional[Dict] = None
) -> str:
    """
    Build a natural-language query for the RAG pipeline based on the problem type.
    Includes actual metrics from the model JSON outputs.
    """
    request = "Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise"

    # --- Build disorder queries using real JSON data ---
    if problem_index == "stuttering" and disorder_result:
        data = disorder_result.get("stuttering", {})
        flagged = data.get("flagged_chunks", [])
        problem = (
            f"The user stutters when speaking. "
            f"Stuttering was detected in {data.get('severity_percentage', 0)}% of the speech, "
            f"specifically in chunks {flagged}."
        )

    elif problem_index == "dysathria" and disorder_result:
        data = disorder_result.get("slurring", {})
        flagged = data.get("flagged_chunks", [])
        problem = (
            f"The user shows signs of dysarthria (slurred speech). "
            f"Slurred speech was detected in {data.get('dysarthric_ratio', 0)}% of the speech, "
            f"specifically in chunks {flagged}."
        )

    # --- Build emotion queries using real JSON data ---
    elif problem_index == "nervouseness" and emotion_result:
        dominant = emotion_result.get("dominant_emotion", "unknown")
        confidence = emotion_result.get("confidence", 0)
        top_3 = emotion_result.get("top_3_emotions", [])
        top_3_str = ", ".join([f"{e['emotion']} ({e['confidence']}%)" for e in top_3])
        problem = (
            f"The user shows nervousness and anxiety when speaking. "
            f"The dominant emotion detected is '{dominant}' with {confidence}% confidence. "
            f"Top 3 emotions: {top_3_str}."
        )

    elif problem_index == "negative_emotion" and emotion_result:
        dominant = emotion_result.get("dominant_emotion", "unknown")
        confidence = emotion_result.get("confidence", 0)
        top_3 = emotion_result.get("top_3_emotions", [])
        top_3_str = ", ".join([f"{e['emotion']} ({e['confidence']}%)" for e in top_3])
        problem = (
            f"The user's speech conveys negative emotion. "
            f"The dominant emotion detected is '{dominant}' with {confidence}% confidence. "
            f"Top 3 emotions: {top_3_str}."
        )

    elif problem_index == "monotone_voice" and emotion_result:
        dominant = emotion_result.get("dominant_emotion", "unknown")
        confidence = emotion_result.get("confidence", 0)
        problem = (
            f"The user speaks in a monotone voice with little pitch variation. "
            f"The dominant emotion detected is '{dominant}' with {confidence}% confidence, "
            f"suggesting flat or low expressiveness."
        )

    elif problem_index == "emotion_inconsistency" and emotion_result:
        top_3 = emotion_result.get("top_3_emotions", [])
        top_3_str = ", ".join([f"{e['emotion']} ({e['confidence']}%)" for e in top_3])
        spread = top_3[0]["confidence"] - top_3[1]["confidence"] if len(top_3) >= 2 else 0
        problem = (
            f"The user's vocal emotion is inconsistent — multiple emotions are competing. "
            f"Top 3 emotions: {top_3_str}. "
            f"The spread between the top 2 emotions is only {round(spread, 1)}%, "
            f"indicating mixed emotional signals."
        )

    # --- Build bad-habit queries using real JSON data ---
    elif problem_index == "filler_words" and bad_habit_result:
        data = bad_habit_result.get("filler_detection", {})
        problem = (
            f"The user uses excessive filler words (um, uh, like) when speaking. "
            f"Filler words were detected in {data.get('filler_rate_pct', 0)}% of the speech, "
            f"with an average filler confidence of {data.get('avg_filler_confidence_pct', 0)}%."
        )

    elif problem_index == "fast_speaking_rate" and bad_habit_result:
        data = bad_habit_result.get("pace_analysis", {})
        problem = (
            f"The user speaks at an unstable or overly fast pace. "
            f"Pace instability was detected in {data.get('instability_rate_pct', 0)}% of the speech, "
            f"with an average instability confidence of {data.get('avg_unstable_confidence_pct', 0)}%."
        )

    else:
        problem = f"The user has a speech problem: {problem_index}"

    query = f"User problem is: {problem}."
    if scenario:
        query += f" The speaking scenario is: {scenario}."
    query += f" The users request is: {request}."

    return query


def _generate_rag_response(
    problem_index: str,
    scenario: str,
    disorder_result: Optional[Dict],
    emotion_result: Optional[Dict],
    bad_habit_result: Optional[Dict] = None,
) -> str:
    """Run RAG for a single problem and return the response text."""
    index_path = os.path.join(INDEX_BASE_DIR, problem_index)

    if not os.path.exists(index_path):
        return f"No knowledge base found for {problem_index}"

    query = _build_query(problem_index, scenario, disorder_result=disorder_result, emotion_result=emotion_result, bad_habit_result=bad_habit_result)

    if problem_index in EMOTION_PROBLEMS:
        instructions = get_emotion_instruction(scenario)
    else:
        instructions = CRITICAL_INSTRUCTION
    try:
        rag_result = run_rag(index_path, query, critical_instructions=instructions)
        response_text = rag_result.get("response", "")
        return response_text.replace("{", "").replace("}", "")
    except Exception as e:
        return f"Error generating feedback: {str(e)}"


def generate_feedback(
    emotion_result: Optional[Dict] = None,
    disorder_result: Optional[Dict] = None,
    bad_habit_result: Optional[Dict] = None,
    scenario: str = "",
) -> Dict:
    """
    Take the JSON outputs from the emotion, disorder, and bad-habit models,
    identify detected problems, generate RAG-based feedback for each,
    and embed the feedback back into the original JSONs.

    Args:
        emotion_result: JSON output from the emotion detection API (/predict)
        disorder_result: JSON output from the disorder detection API (/api/analyze/disorder).
                         This should be the 'data' field from the API response.
        bad_habit_result: JSON output from the bad-habit detection API (/analyze)
        scenario: Optional speaking scenario (e.g. "university viva", "job interview")

    Returns:
        Dictionary with:
            - emotion_result: original emotion JSON with "rag_feedback" added per problem
            - disorder_result: original disorder JSON with "rag_feedback" added per flagged disorder
            - bad_habit_result: original bad-habit JSON with "rag_feedback" added per flagged habit
    """
    problems = _identify_problems(emotion_result, disorder_result, bad_habit_result)

    # Deep copy so we don't mutate the originals
    emotion_out = dict(emotion_result) if emotion_result else None
    disorder_out = dict(disorder_result) if disorder_result else None
    bad_habit_out = dict(bad_habit_result) if bad_habit_result else None

    if not problems:
        if emotion_out:
            emotion_out["rag_feedback"] = {}
        if disorder_out:
            disorder_out["stuttering"]["rag_feedback"] = None
            disorder_out["slurring"]["rag_feedback"] = None
        if bad_habit_out:
            bad_habit_out["filler_detection"]["rag_feedback"] = None
            bad_habit_out["pace_analysis"]["rag_feedback"] = None
        return {"emotion_result": emotion_out, "disorder_result": disorder_out, "bad_habit_result": bad_habit_out}

    # --- Generate feedback for each problem and embed into the right JSON ---

    # Emotion problems: store under emotion_result["rag_feedback"][problem_name]
    emotion_feedback = {}
    emotion_problems = [p for p in problems if p in EMOTION_PROBLEM_MAP.values()]
    for problem_index in emotion_problems:
        response = _generate_rag_response(problem_index, scenario, disorder_result, emotion_result)
        emotion_feedback[problem_index] = response

    if emotion_out:
        emotion_out["rag_feedback"] = emotion_feedback

    # Disorder problems: store under disorder_result["stuttering"]["rag_feedback"] etc.
    if disorder_out:
        if "stuttering" in problems:
            response = _generate_rag_response("stuttering", scenario, disorder_out, emotion_result)
            disorder_out["stuttering"]["rag_feedback"] = response
        else:
            disorder_out["stuttering"]["rag_feedback"] = None

        if "dysathria" in problems:
            response = _generate_rag_response("dysathria", scenario, disorder_out, emotion_result)
            disorder_out["slurring"]["rag_feedback"] = response
        else:
            disorder_out["slurring"]["rag_feedback"] = None

    # Bad-habit problems: store under bad_habit_result["filler_detection"]["rag_feedback"] etc.
    if bad_habit_out:
        if "filler_words" in problems:
            response = _generate_rag_response("filler_words", scenario, disorder_result, emotion_result, bad_habit_result=bad_habit_out)
            bad_habit_out["filler_detection"]["rag_feedback"] = response
        else:
            bad_habit_out["filler_detection"]["rag_feedback"] = None

        if "fast_speaking_rate" in problems:
            response = _generate_rag_response("fast_speaking_rate", scenario, disorder_result, emotion_result, bad_habit_result=bad_habit_out)
            bad_habit_out["pace_analysis"]["rag_feedback"] = response
        else:
            bad_habit_out["pace_analysis"]["rag_feedback"] = None

    return {"emotion_result": emotion_out, "disorder_result": disorder_out, "bad_habit_result": bad_habit_out}


