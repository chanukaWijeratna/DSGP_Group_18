"""
====================================================================
  Emotion Detection API — Multi-modal Speech Analysis & Feedback BOT
  Component   : Emotion Detection
  Author      : [Your Name] | Group 18
  Framework   : FastAPI
  Model Input : MFCC Features — CNN (.h5 format)
  Datasets    : RAVDESS + CREMA-D
  Emotions    : angry, calm, disgust, fearful, happy, neutral, sad, surprised
                (alphabetical order — matches sklearn LabelEncoder training)
====================================================================

TO RUN (in VS Code terminal):
    1. Activate your virtual environment:
       venv\Scripts\activate

    2. Start the server:
       uvicorn emotion_detection_api:app --reload --host 0.0.0.0 --port 8000

    3. API will be live at:
       http://localhost:8000

    4. Interactive docs (Swagger UI — test file uploads here):
       http://localhost:8000/docs

ENDPOINTS:
    POST /predict     — Upload audio file → returns emotion analysis JSON
    GET  /health      — Check if API is running
    GET  /            — Welcome message
====================================================================
"""

import os
import uuid
import tempfile
import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model

# ──────────────────────────────────────────────
#  Configuration — matches your training setup
#  exactly as defined in your notebook
# ──────────────────────────────────────────────

MODEL_PATH      = os.path.join(os.path.dirname(__file__), "..", "models", "emotion_model.h5")
SAMPLE_RATE     = 22050                # Matches librosa.load sr in your notebook
N_MFCC          = 40                   # Matches n_mfcc=40 in your notebook
MAX_PAD_LENGTH  = 130                  # Matches model input shape (None, 40, 130, 1)
SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
FRAME_DURATION    = 3.0    # seconds per frame
# Dynamic thresholds per problem category — scales with audio length
# Longer clips need a lower % of frames because even a small fraction
# represents a significant absolute duration of detected distress.
# Format: list of (max_frame_count, threshold) checked in order.
PROBLEM_THRESHOLDS = {
    "Stress/Anxiety Tone": [
        # Intermittent by nature — nervous speakers spike then recover
        (5,  0.50),   # ≤15 s  → need half the frames (guard against noise)
        (20, 0.40),   # ≤1 min → 40 %
        (40, 0.30),   # ≤2 min → 30 % (~36 s of stress in 2 min)
        (None, 0.20), # >2 min → 20 % (~36 s in 3 min — still very notable)
    ],
    "Negative Emotional Dominance": [
        # Sustained negativity — still important at lower ratios in long clips
        (5,  0.55),
        (20, 0.45),
        (40, 0.35),
        (None, 0.25),
    ],
    "Flat/Low Expressiveness": [
        # Monotone speech tends to be persistent; need a higher bar
        (5,  0.65),
        (20, 0.55),
        (40, 0.45),
        (None, 0.35),
    ],
    "Emotion Inconsistency": [
        # Mixed signals can be brief but meaningful
        (5,  0.50),
        (20, 0.40),
        (40, 0.30),
        (None, 0.20),
    ],
}
MIN_CONFIDENCE    = 50.0   # skip frames where model confidence is below this

# ── Exact label order from your LabelEncoder ──
# sklearn LabelEncoder.fit() sorts alphabetically
# Verified from: RAVDESS + CREMA-D emotion mappings in your notebook
# RAVDESS: neutral, calm, happy, sad, angry, fearful, disgust, surprised
# CREMA-D: angry, disgust, fearful, happy, neutral, sad
# Combined & sorted → the 8 labels below
EMOTION_LABELS = [
    "angry",      # 0
    "calm",       # 1
    "disgust",    # 2
    "fearful",    # 3
    "happy",      # 4
    "neutral",    # 5
    "sad",        # 6
    "surprised"   # 7
]

# ── Problem descriptions → for the Feedback BOT ──
# Maps to the 5 emotion problems defined in your notebook:
# 1. Flat/Low Expressiveness      → neutral with low energy
# 2. Stress/Anxiety Tone          → fearful, angry
# 3. Negative Emotional Dominance → angry, disgust, sad, fearful
# 4. Emotion Inconsistency        → detected via top_3 spread
# 5. Intended vs Perceived        → flagged when confidence is low
PROBLEM_MAP = {
    "angry"    : "Stress/Anxiety Tone — High anger or frustration detected in speech.",
    "calm"     : "No significant emotional problem — calm and composed tone detected.",
    "disgust"  : "Negative Emotional Dominance — Strong disgust or disapproval detected.",
    "fearful"  : "Stress/Anxiety Tone — Fear or anxiety indicators detected in speech.",
    "happy"    : "No significant emotional problem — positive and happy tone detected.",
    "neutral"  : "No significant emotional problem — neutral and steady tone detected.",
    "sad"      : "Negative Emotional Dominance — Sadness or low energy detected in speech.",
    "surprised": "No significant emotional problem — expressive and surprised tone detected.",
}

# Emotions that should NOT be flagged as problems on their own
NON_PROBLEM_EMOTIONS = {"calm", "happy", "neutral", "surprised"}

# ──────────────────────────────────────────────
#  Load model at startup
# ──────────────────────────────────────────────

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\n[ERROR] Model file '{MODEL_PATH}' not found.\n"
        f"Please place your trained model in the same folder as this script.\n"
    )

print(f"[INFO] Loading model from '{MODEL_PATH}' ...")
model = load_model(MODEL_PATH)
print(f"[INFO] Model loaded successfully.")
print(f"[INFO] Input shape : {model.input_shape}")
print(f"[INFO] Output shape: {model.output_shape}")
print(f"[INFO] Emotion labels: {EMOTION_LABELS}")

# ──────────────────────────────────────────────
#  FastAPI app setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="Emotion Detection API",
    description=(
        "Speech Emotion Detection component of the Multi-modal Speech Analysis "
        "and Feedback BOT project (Group 18). Accepts audio files and returns "
        "detected emotions and problems for the Feedback BOT."
    ),
    version="1.0.0",
)

# CORS — allows web app and mobile app to call this API freely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
#  Helper functions
# ──────────────────────────────────────────────

def extract_mfcc_from_segment(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract MFCC features from a pre-loaded audio segment.
    Output shape: (40, 130) — padded/truncated to MAX_PAD_LENGTH
    """
    expected_length = int(sr * FRAME_DURATION)
    if len(audio) < expected_length:
        audio = np.pad(audio, (0, expected_length - len(audio)))

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_PAD_LENGTH:
        pad_width = MAX_PAD_LENGTH - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_PAD_LENGTH]

    return mfcc  # shape: (40, 130)


def extract_frames(file_path: str) -> list:
    """
    Load the full audio and split into FRAME_DURATION-second frames.
    Returns a list of MFCC arrays, one per frame.
    """
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    frame_length = int(sr * FRAME_DURATION)
    frames = []

    for start in range(0, len(audio), frame_length):
        segment = audio[start : start + frame_length]
        # skip frames shorter than 1 second — not enough signal
        if len(segment) < int(sr * 1.0):
            continue
        frames.append(extract_mfcc_from_segment(segment, sr))

    return frames


def get_problem_category(problem: str) -> str:
    """Extract the problem category from a full problem description string."""
    for category in PROBLEM_THRESHOLDS:
        if problem.startswith(category):
            return category
    return ""


def get_threshold_for_problem(problem: str, frame_count: int) -> float:
    """Return the dynamic threshold for a problem based on audio length."""
    category = get_problem_category(problem)
    tiers = PROBLEM_THRESHOLDS.get(category)
    if not tiers:
        return 0.50  # fallback for unrecognised problems

    for max_frames, threshold in tiers:
        if max_frames is None or frame_count <= max_frames:
            return threshold
    return tiers[-1][1]


def aggregate_results(frame_results: list) -> dict:
    """
    Aggregate per-frame predictions across the full audio.

    - dominant_emotion : most frequent emotion across all frames
    - confidence       : average confidence of the dominant emotion
    - top_3_emotions   : averaged probabilities across all frames
    - detected_problem : problems that appeared in >PROBLEM_THRESHOLD of frames
    - frame_count      : total number of frames analysed
    """
    total = len(frame_results)

    # average probability per emotion across all frames
    avg_probs = np.mean([r["probabilities"] for r in frame_results], axis=0)
    dominant_idx     = int(np.argmax(avg_probs))
    dominant_emotion = EMOTION_LABELS[dominant_idx]
    confidence       = round(float(avg_probs[dominant_idx]) * 100, 2)
    top_3            = get_top_3_emotions(avg_probs)

    # count how many frames flagged each problem
    problem_counts = {}
    for r in frame_results:
        p = r["problem"]
        problem_counts[p] = problem_counts.get(p, 0) + 1

    # only keep problems that exceeded their dynamic threshold,
    # excluding "no problem" entries
    flagged_problems = [
        p for p, count in problem_counts.items()
        if count / total >= get_threshold_for_problem(p, total)
        and "No significant emotional problem" not in p
    ]

    if not flagged_problems:
        fallback = detect_problem(dominant_emotion, top_3, confidence)
        flagged_problems = [fallback]

    return {
        "dominant_emotion" : dominant_emotion,
        "confidence"       : confidence,
        "top_3_emotions"   : top_3,
        "detected_problems": flagged_problems,
        "frame_count"      : total,
        "problem_counts"   : {p: round(c / total * 100, 1) for p, c in problem_counts.items()},
    }


def prepare_model_input(mfcc: np.ndarray) -> np.ndarray:
    """
    Reshape MFCC to match CNN input: (1, 40, 130, 1)
    Matches X_train_mfcc = X_train_mfcc[..., np.newaxis] in your notebook
    """
    return mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1)


def get_top_3_emotions(probabilities: np.ndarray) -> list:
    """
    Return top 3 emotions with confidence scores (sorted highest first).
    """
    top_indices = np.argsort(probabilities)[::-1][:3]
    return [
        {
            "emotion"   : EMOTION_LABELS[i],
            "confidence": round(float(probabilities[i]) * 100, 2)
        }
        for i in top_indices
    ]


def detect_problem(dominant_emotion: str, top_3: list, confidence: float) -> str:
    """
    Maps detected emotion to one of the 4 emotion problems from your notebook:
    1. Flat/Low Expressiveness
    2. Stress/Anxiety Tone
    3. Negative Emotional Dominance
    4. Emotion Inconsistency
    """
    negative = {"angry", "disgust", "fearful", "sad"}
    top_names = [e["emotion"] for e in top_3]
    neg_count = sum(1 for e in top_names if e in negative)

    # Problem 1: Flat/Low Expressiveness — neutral dominates with very high confidence
    # Only flag when the speaker is overwhelmingly monotone (>80% neutral)
    if dominant_emotion == "neutral" and confidence > 80.0:
        return (
            "Flat/Low Expressiveness — speech is overwhelmingly neutral with minimal "
            "emotional variation. Speaker may benefit from more vocal expressiveness."
        )

    # Problem 3: Negative Emotional Dominance
    # Only flag if the DOMINANT emotion is negative AND it has strong confidence,
    # OR if all top 3 are negative (clear pattern)
    if dominant_emotion in negative and neg_count >= 3:
        labels = " and ".join([e for e in top_names if e in negative])
        return (
            f"Negative Emotional Dominance — multiple negative emotions detected "
            f"({labels}). Speaker may be under significant emotional distress."
        )

    # Problem 4: Emotion Inconsistency — top 2 emotions too close
    # Only flag if at least one of the competing emotions is negative
    if len(top_3) >= 2:
        spread = top_3[0]["confidence"] - top_3[1]["confidence"]
        if spread < 10.0 and any(e in negative for e in top_names[:2]):
            return (
                "Emotion Inconsistency — two or more emotions are closely competing. "
                "Speech may contain mixed emotional signals."
            )

    # If dominant emotion is not a problem emotion, return no-problem
    if dominant_emotion in NON_PROBLEM_EMOTIONS:
        return PROBLEM_MAP.get(dominant_emotion, "No significant emotional problem detected.")

    return PROBLEM_MAP.get(dominant_emotion, "Emotion pattern unclear — further analysis recommended.")


# ──────────────────────────────────────────────
#  API Endpoints
# ──────────────────────────────────────────────

@app.get("/", summary="Welcome")
def root():
    return {
        "message"   : "Emotion Detection API is running.",
        "project"   : "Multi-modal Speech Analysis and Feedback BOT",
        "component" : "Emotion Detection — Group 18",
        "docs"      : "/docs",
        "version"   : "1.0.0"
    }


@app.get("/health", summary="Health Check")
def health_check():
    """
    Health check endpoint — used by the Feedback BOT and frontend
    to verify this API is live before sending requests.
    """
    return {
        "status"            : "healthy",
        "model_loaded"      : True,
        "model_input_shape" : str(model.input_shape),
        "emotion_labels"    : EMOTION_LABELS,
        "num_classes"       : len(EMOTION_LABELS),
        "supported_formats" : list(SUPPORTED_FORMATS),
    }


@app.post("/predict", summary="Predict Emotion from Audio")
async def predict_emotion(file: UploadFile = File(...)):
    """
    Upload an audio file to detect the speaker's emotion.

    Accepted formats: .wav, .mp3, .flac, .ogg, .m4a

    Returns a JSON with:
    - dominant_emotion  : top detected emotion
    - confidence        : confidence % of dominant emotion
    - top_3_emotions    : top 3 emotions with confidence scores
    - detected_problem  : mapped emotion problem (for Feedback BOT)
    - summary_note      : human-readable summary for Feedback BOT
    """

    # ── 1. Validate file format ──────────────────
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[-1].lower()

    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format '{ext}'. "
                f"Accepted formats: {', '.join(SUPPORTED_FORMATS)}"
            )
        )

    # ── 2. Save upload to a temporary file ───────
    temp_path = os.path.join(tempfile.gettempdir(), f"emotion_{uuid.uuid4().hex}{ext}")

    try:
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)

        # ── 3. Split audio into frames & extract MFCCs ──
        try:
            frames = extract_frames(temp_path)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Audio processing failed: {str(e)}. Please ensure the file contains valid audio."
            )

        if not frames:
            raise HTTPException(
                status_code=422,
                detail="Audio is too short to analyse. Please provide at least 1 second of audio."
            )

        # ── 4. Run CNN inference on each frame ───────
        try:
            frame_results = []
            skipped_frames = 0
            for mfcc in frames:
                model_input  = prepare_model_input(mfcc)
                probs        = model.predict(model_input, verbose=0)[0]  # shape: (8,)
                dom_idx      = int(np.argmax(probs))
                dom_conf     = round(float(probs[dom_idx]) * 100, 2)

                # skip frame if model is uncertain — likely a model/audio issue, not user problem
                if dom_conf < MIN_CONFIDENCE:
                    skipped_frames += 1
                    continue

                dom_emotion  = EMOTION_LABELS[dom_idx]
                top_3        = get_top_3_emotions(probs)
                problem      = detect_problem(dom_emotion, top_3, dom_conf)
                frame_results.append({
                    "probabilities": probs,
                    "emotion"      : dom_emotion,
                    "confidence"   : dom_conf,
                    "problem"      : problem,
                })
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Model inference error: {str(e)}"
            )

        if not frame_results:
            raise HTTPException(
                status_code=422,
                detail="Could not confidently detect any emotion in the audio. Please try a clearer recording."
            )

        # ── 5. Aggregate across all frames ───────────
        agg = aggregate_results(frame_results)

        summary_note = (
            f"Analysed {agg['frame_count']} frames ({skipped_frames} skipped — low model confidence). "
            f"Dominant emotion: '{agg['dominant_emotion'].upper()}' "
            f"({agg['confidence']}% avg confidence). "
            f"Flagged problems: {'; '.join(agg['detected_problems'])}"
        )

        # ── 6. Return JSON for Feedback BOT ──────────
        return JSONResponse(content={
            "status"            : "success",
            "file_name"         : filename,
            "frame_count"       : agg["frame_count"],
            "skipped_frames"    : skipped_frames,
            "dominant_emotion"  : agg["dominant_emotion"],
            "confidence"        : agg["confidence"],
            "top_3_emotions"    : agg["top_3_emotions"],
            "detected_problems" : agg["detected_problems"],
            "problem_counts"    : agg["problem_counts"],
            "summary_note"      : summary_note,
        })

    finally:
        # Always clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ──────────────────────────────────────────────
#  Run directly (python emotion_detection_api.py)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("emotion_detection_api:app", host="0.0.0.0", port=8000, reload=True)