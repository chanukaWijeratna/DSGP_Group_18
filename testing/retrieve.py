import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDINGS_DIR = "embeddings"

configs = [
    # --- MiniLM-L6-v2, chunk=256, threshold=0.3 — top_k sweep ---
    {"name": "R1",  "embedding": "all-MiniLM-L6-v2",  "chunk_size": 256, "top_k": 1,  "threshold": 0.3},
    {"name": "R2",  "embedding": "all-MiniLM-L6-v2",  "chunk_size": 256, "top_k": 3,  "threshold": 0.3},
    {"name": "R3",  "embedding": "all-MiniLM-L6-v2",  "chunk_size": 256, "top_k": 5,  "threshold": 0.3},
    {"name": "R4",  "embedding": "all-MiniLM-L6-v2",  "chunk_size": 256, "top_k": 7,  "threshold": 0.3},
    {"name": "R5",  "embedding": "all-MiniLM-L6-v2",  "chunk_size": 256, "top_k": 10, "threshold": 0.3},
    # --- MiniLM-L6-v2, chunk=512, threshold=0.3 — top_k sweep ---
    {"name": "R6",  "embedding": "all-MiniLM-L6-v2",  "chunk_size": 512, "top_k": 1,  "threshold": 0.3},
    {"name": "R7",  "embedding": "all-MiniLM-L6-v2",  "chunk_size": 512, "top_k": 3,  "threshold": 0.3},
    {"name": "R8",  "embedding": "all-MiniLM-L6-v2",  "chunk_size": 512, "top_k": 5,  "threshold": 0.3},
    {"name": "R9",  "embedding": "all-MiniLM-L6-v2",  "chunk_size": 512, "top_k": 7,  "threshold": 0.3},
    {"name": "R10", "embedding": "all-MiniLM-L6-v2",  "chunk_size": 512, "top_k": 10, "threshold": 0.3},
    # --- mpnet-base-v2, chunk=256, threshold=0.5 — top_k sweep ---
    {"name": "R11", "embedding": "all-mpnet-base-v2", "chunk_size": 256, "top_k": 1,  "threshold": 0.5},
    {"name": "R12", "embedding": "all-mpnet-base-v2", "chunk_size": 256, "top_k": 3,  "threshold": 0.5},
    {"name": "R13", "embedding": "all-mpnet-base-v2", "chunk_size": 256, "top_k": 5,  "threshold": 0.5},
    {"name": "R14", "embedding": "all-mpnet-base-v2", "chunk_size": 256, "top_k": 7,  "threshold": 0.5},
    {"name": "R15", "embedding": "all-mpnet-base-v2", "chunk_size": 256, "top_k": 10, "threshold": 0.5},
    # --- mpnet-base-v2, chunk=512, threshold=0.5 — top_k sweep ---
    {"name": "R16", "embedding": "all-mpnet-base-v2", "chunk_size": 512, "top_k": 1,  "threshold": 0.5},
    {"name": "R17", "embedding": "all-mpnet-base-v2", "chunk_size": 512, "top_k": 3,  "threshold": 0.5},
    {"name": "R18", "embedding": "all-mpnet-base-v2", "chunk_size": 512, "top_k": 5,  "threshold": 0.5},
    {"name": "R19", "embedding": "all-mpnet-base-v2", "chunk_size": 512, "top_k": 7,  "threshold": 0.5},
    {"name": "R20", "embedding": "all-mpnet-base-v2", "chunk_size": 512, "top_k": 10, "threshold": 0.5},
]

test_queries = [
    # === FILLER WORDS (6 queries) ===
    {
        "query": "User problem is: Excessive filler words detected. Filler count: 18 instances of 'um', 'uh', 'like' in a 2-minute speech. Filler rate: 9 per minute. The speaking scenario is: Job interview practice. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["filler_words.md"]
    },
    {
        "query": "User problem is: Moderate filler word usage detected. Filler count: 8 instances of 'you know', 'actually', 'so' in a 3-minute speech. The speaking scenario is: University viva presentation. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["filler_words.md"]
    },
    {
        "query": "User problem is: High filler word frequency detected. 'Um' appeared 12 times, 'uh' appeared 7 times. Total words spoken: 250. The speaking scenario is: Team meeting at work. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["filler_words.md"]
    },
    {
        "query": "User problem is: Filler words detected throughout speech. Repeated use of 'like' and 'you know' totaling 22 instances in 4 minutes. The speaking scenario is: Casual public speaking event. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["filler_words.md"]
    },
    {
        "query": "User problem is: Mild filler word usage. 5 instances of 'um' and 'so' detected in a 2-minute recording. Filler rate: 2.5 per minute. The speaking scenario is: Classroom presentation. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["filler_words.md"]
    },
    {
        "query": "User problem is: Frequent filler word clusters detected. Multiple back-to-back fillers 'um um like' found in transitions between points. Total filler count: 15. The speaking scenario is: Conference talk rehearsal. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["filler_words.md"]
    },

    # === MONOTONE VOICE (4 queries) ===
    {
        "query": "User problem is: Monotone speech detected. Pitch variance is very low at 12Hz across the entire recording. Minimal intonation changes observed. The speaking scenario is: Online lecture delivery. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["monotone_voice.md"]
    },
    {
        "query": "User problem is: Flat pitch pattern detected. Speaker maintains nearly constant pitch throughout a 3-minute speech with no emphasis variation. The speaking scenario is: Sales pitch to a client. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["monotone_voice.md"]
    },
    {
        "query": "User problem is: Low pitch variation detected. Pitch standard deviation is below threshold. Speech sounds robotic and lacks expressive intonation. The speaking scenario is: Wedding toast practice. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["monotone_voice.md"]
    },
    {
        "query": "User problem is: Monotone delivery detected. Pitch range is extremely narrow. No rise or fall in intonation at sentence boundaries. The speaking scenario is: YouTube video recording. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["monotone_voice.md"]
    },

    # === FAST SPEAKING RATE (4 queries) ===
    {
        "query": "User problem is: Speaking pace too fast. Average speaking rate: 195 words per minute. Rate is unstable with significant variation across segments. The speaking scenario is: Thesis defense presentation. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["fast_speaking_rate.md"]
    },
    {
        "query": "User problem is: Rapid speaking rate detected. 210 WPM average with minimal pausing between sentences. Speech feels rushed. The speaking scenario is: Customer service call. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["fast_speaking_rate.md"]
    },
    {
        "query": "User problem is: Unstable and fast speaking pace. Rate fluctuates between 160-220 WPM across segments. Audience may struggle to follow. The speaking scenario is: Debate competition. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["fast_speaking_rate.md"]
    },
    {
        "query": "User problem is: Excessively fast delivery detected. Speaking rate: 205 WPM. Words are being clipped and sentences run together. The speaking scenario is: Radio podcast recording. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["fast_speaking_rate.md"]
    },

    # === STUTTERING (4 queries) ===
    {
        "query": "User problem is: Stuttering events detected. 6 repetitions and 2 blocks identified in a 2-minute speech sample. Possible speech disfluency pattern. The speaking scenario is: Office team standup meeting. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["stuttering.md"]
    },
    {
        "query": "User problem is: Speech disfluency detected. 4 prolongation events and 3 word repetitions found. Disfluency rate is moderate. The speaking scenario is: Phone call with a client. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["stuttering.md"]
    },
    {
        "query": "User problem is: Repetition-type stuttering detected. Speaker repeated initial syllables 8 times in 3 minutes. Pattern is consistent at sentence beginnings. The speaking scenario is: Classroom question-answer session. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["stuttering.md"]
    },
    {
        "query": "User problem is: Blocking events detected in speech. 5 instances where speech halted mid-sentence before continuing. Blocks lasted 1-2 seconds each. The speaking scenario is: Job interview. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["stuttering.md"]
    },

    # === DYSARTHRIA (3 queries) ===
    {
        "query": "User problem is: Possible slurred speech detected. Articulation clarity score is low. Words are not clearly pronounced and consonants are weak. The speaking scenario is: Video call presentation at work. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["dysathria.md"]
    },
    {
        "query": "User problem is: Speech slurring detected with high confidence. Unclear articulation across multiple segments. Speech sounds muffled. The speaking scenario is: Public speaking event. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["dysathria.md"]
    },
    {
        "query": "User problem is: Articulation issues detected. Consonant sounds are consistently weak and vowel sounds are distorted. Possible motor speech difficulty. The speaking scenario is: University group presentation. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["dysathria.md"]
    },

    # === EMOTION INCONSISTENCY (4 queries) ===
    {
        "query": "User problem is: Emotion inconsistency detected. Predicted emotion: anger with high confidence. Energy variance across speech segments is very high. Tone does not match expected delivery. The speaking scenario is: Formal business presentation. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["emotion_inconsistency.md"]
    },
    {
        "query": "User problem is: Emotional tone mismatch detected. Speech conveys sadness but context requires confident delivery. Confidence score: 0.82 for sad emotion. The speaking scenario is: Motivational speech practice. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["emotion_inconsistency.md"]
    },
    {
        "query": "User problem is: Emotion inconsistency detected. Speech switches between fearful and neutral tones multiple times. Energy levels fluctuate significantly across segments. The speaking scenario is: Investor pitch. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["emotion_inconsistency.md"]
    },
    {
        "query": "User problem is: Unexpected emotional tone detected. Predicted emotion: disgust with moderate confidence. Vocal energy pattern shows inconsistency. The speaking scenario is: Teaching a class. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["emotion_inconsistency.md"]
    },

    # === NERVOUSNESS (3 queries) ===
    {
        "query": "User problem is: Signs of nervousness detected in speech. Predicted emotion: fear with high confidence. Voice is shaky and pitch is higher than normal range. The speaking scenario is: First-time conference presentation. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["nervouseness.md"]
    },
    {
        "query": "User problem is: Nervous speech patterns detected. Rapid breathing pauses, rising pitch at sentence ends, and trembling vocal quality observed. The speaking scenario is: PhD viva voce examination. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["nervouseness.md"]
    },
    {
        "query": "User problem is: Anxiety indicators found in speech. Voice tremor detected, uneven breathing pattern, and frequent throat clearing. Predicted emotion: fearful. The speaking scenario is: Award acceptance speech. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["nervouseness.md"]
    },

    # === NEGATIVE EMOTION (5 queries) ===
    {
        "query": "User problem is: Negative emotion dominance detected in speech. Vocal tone conveys anger and frustration despite neutral content. Pitch is elevated and intensity is high. The speaking scenario is: Team meeting at work. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["negative_emotion.md"]
    },
    {
        "query": "User problem is: Speech tone analysis reveals strong negative emotional cues. Voice sounds tense and sharp with increased vocal intensity. Listeners may perceive hostility. The speaking scenario is: Client negotiation call. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["negative_emotion.md"]
    },
    {
        "query": "User problem is: Emotional tone mismatch with intended message. Speaker's voice carries irritation and defensiveness. Vocal tension and rapid speech rate indicate emotional arousal. The speaking scenario is: Performance review discussion. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["negative_emotion.md"]
    },
    {
        "query": "User problem is: Negative vocal affect detected. Speech sounds strained and confrontational. Pitch range is narrow and intensity spikes during key phrases. Emotional regulation appears low. The speaking scenario is: Formal business presentation. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["negative_emotion.md"]
    },
    {
        "query": "User problem is: Frustration and anxiety dominating speech delivery. Voice is tight with uneven breathing patterns. Emotional cues override the constructive intent of the message. The speaking scenario is: Parent-teacher conference. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["negative_emotion.md"]
    },

    # === ADDITIONAL SINGLE-PROBLEM QUERIES (2 queries) ===
    {
        "query": "User problem is: Nervous vocal quality detected. Speaker's voice pitch is elevated and unstable. Breathing is shallow and irregular throughout the recording. The speaking scenario is: Toastmasters club speech. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["nervouseness.md"]
    },
    {
        "query": "User problem is: Slurred articulation detected. Speech clarity is poor with consonants being dropped and words blending together. Confidence score is high. The speaking scenario is: Online job interview. The users request is: Explain the problem, describe what reasons behind it, provide simple, actionable improvement advice with a quick practice exercise.",
        "expected_files": ["dysathria.md"]
    }
]

def load_config_data(embedding_model: str, chunk_size: int):
    """Load all embeddings and chunks for a given model/chunk_size config."""
    folder = f"{embedding_model}_chunk{chunk_size}"
    folder_path = os.path.join(EMBEDDINGS_DIR, folder)

    all_embeddings = []
    all_chunks = []

    metadata_files = sorted(f for f in os.listdir(folder_path) if f.endswith("_metadata.json"))
    for meta_file in metadata_files:
        with open(os.path.join(folder_path, meta_file), "r", encoding="utf-8") as f:
            metadata = json.load(f)

        npy_file = meta_file.replace("_metadata.json", "_embeddings.npy")
        embeddings = np.load(os.path.join(folder_path, npy_file))

        for i, chunk in enumerate(metadata["chunks"]):
            all_chunks.append({
                "text": chunk,
                "source": metadata["source_file"],
                "chunk_index": i,
            })
        all_embeddings.append(embeddings)

    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings, all_chunks


def retrieve(query: str, config: dict, model: SentenceTransformer, embeddings: np.ndarray, chunks: list):
    """Retrieve top_k chunks above the similarity threshold for a query."""
    query_embedding = model.encode([query])[0]

    # Cosine similarity (embeddings from SentenceTransformer are already normalized)
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    # Filter by threshold, then take top_k
    indices = np.where(similarities >= config["threshold"])[0]
    indices = indices[np.argsort(similarities[indices])[::-1]][:config["top_k"]]

    results = []
    for idx in indices:
        results.append({
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "chunk_index": chunks[idx]["chunk_index"],
            "similarity": float(similarities[idx]),
        })

    return results


def retrieve_all_configs(query: str):
    """Run retrieval across all configs and return results keyed by config name."""
    # Cache models to avoid reloading
    models = {}
    config_data = {}

    for cfg in configs:
        model_name = cfg["embedding"]
        chunk_size = cfg["chunk_size"]
        data_key = f"{model_name}_chunk{chunk_size}"

        if model_name not in models:
            models[model_name] = SentenceTransformer(model_name)
        if data_key not in config_data:
            config_data[data_key] = load_config_data(model_name, chunk_size)

    all_results = {}
    for cfg in configs:
        model_name = cfg["embedding"]
        data_key = f"{model_name}_chunk{cfg['chunk_size']}"
        embeddings, chunks = config_data[data_key]
        model = models[model_name]

        results = retrieve(query, cfg, model, embeddings, chunks)
        all_results[cfg["name"]] = {
            "config": cfg,
            "results": results,
        }

    return all_results


if __name__ == "__main__":
    query = "What causes stuttering?"
    all_results = retrieve_all_configs(query)

    for name, data in all_results.items():
        cfg = data["config"]
        results = data["results"]
        print(f"\n{'='*60}")
        print(f"{name}: model={cfg['embedding']}, chunk={cfg['chunk_size']}, top_k={cfg['top_k']}, threshold={cfg['threshold']}")
        print(f"Retrieved {len(results)} chunks")
        print(f"{'='*60}")
        for r in results:
            print(f"  [{r['similarity']:.4f}] {r['source']} (chunk {r['chunk_index']})")
            print(f"    {r['text'][:120]}...")
