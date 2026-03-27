"""
====================================================================
  Bad Habit Detection API — Filler Words + Pace Instability
  Component   : Audio Analysis
  Framework   : FastAPI
  Port        : 8001
====================================================================

TO RUN (from backend/bad-habit-detection/routes/):
    uvicorn analyze_audio:app --host 0.0.0.0 --port 8001

ENDPOINTS:
    POST /analyze   — Accept audio file, return filler + pace analysis
    GET  /health    — Health check
====================================================================
"""

import os
import json
import tempfile

import librosa
import numpy as np
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Constants ──────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000
CHUNK_SEC     = 5
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SEC

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3",
    "audio/ogg", "audio/webm", "audio/flac", "audio/mp4",
}

# Dynamic thresholds per habit — scales with audio length (same pattern as emotion module).
# Longer clips use a lower % because even a small fraction represents significant duration.
# Format: list of (max_chunk_count, threshold) checked in order.
HABIT_THRESHOLDS = {
    "filler": [
        # Filler words are intermittent — speakers spike then recover
        (3,   0.50),    # ≤15 s  → need half the chunks (guard against noise)
        (12,  0.35),    # ≤1 min → 35 %
        (24,  0.25),    # ≤2 min → 25 % (~30 s of filler in 2 min)
        (None, 0.15),   # >2 min → 15 % (still notable at this length)
    ],
    "pace": [
        # Pace instability tends to be more sustained; use a slightly higher bar
        (3,   0.55),    # ≤15 s
        (12,  0.40),    # ≤1 min
        (24,  0.30),    # ≤2 min
        (None, 0.20),   # >2 min
    ],
}

FILLER_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "filler_detector")
PACE_MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "models", "pace_model")

# ── FastAPI app ────────────────────────────────────────────────────
app = FastAPI(
    title="Bad Habit Detection API",
    description="Filler word detection and pace instability analysis for Vocal Insight.",
    version="1.0.0",
)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading (on startup) ────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Device: {device}")
print(f"[INFO] Loading filler model from '{FILLER_MODEL_DIR}' ...")
filler_fe    = AutoFeatureExtractor.from_pretrained(FILLER_MODEL_DIR)
filler_model = AutoModelForAudioClassification.from_pretrained(FILLER_MODEL_DIR).to(device)
filler_model.eval()
print(f"[INFO] Filler model loaded successfully.")

print(f"[INFO] Loading pace model from '{PACE_MODEL_DIR}' ...")
pace_fe    = Wav2Vec2FeatureExtractor.from_pretrained(PACE_MODEL_DIR)
pace_model = Wav2Vec2ForSequenceClassification.from_pretrained(PACE_MODEL_DIR).to(device)
pace_model.eval()
print(f"[INFO] Pace model loaded successfully.")


# ── Audio helpers ──────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    audio    = audio.astype(np.float32)
    peak     = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio


def make_chunks(audio: np.ndarray):
    return [
        audio[s : s + CHUNK_SAMPLES]
        for s in range(0, len(audio) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES)
    ]


# ── Inference ──────────────────────────────────────────────────────

@torch.no_grad()
def run_filler(chunks):
    inp = filler_fe(
        chunks,
        sampling_rate  = SAMPLE_RATE,
        return_tensors = "pt",
        padding        = True,
        max_length     = CHUNK_SAMPLES,
        truncation     = True,
    )
    inp    = {k: v.to(device) for k, v in inp.items()}
    logits = filler_model(**inp).logits
    probs  = torch.softmax(logits, dim=-1).cpu().numpy()

    results = []
    for i, p in enumerate(probs):
        pred  = int(np.argmax(p))
        label = filler_model.config.id2label[pred]
        results.append({
            "chunk_index" : i + 1,
            "time_start_s": i * CHUNK_SEC,
            "time_end_s"  : (i + 1) * CHUNK_SEC,
            "label"       : label,
            "clean_pct"   : round(float(p[0]) * 100, 2),
            "filler_pct"  : round(float(p[1]) * 100, 2),
        })
    return results


@torch.no_grad()
def run_pace(chunks):
    inp = pace_fe(
        chunks,
        sampling_rate  = SAMPLE_RATE,
        return_tensors = "pt",
        padding        = True,
        max_length     = CHUNK_SAMPLES,
        truncation     = True,
    )
    inp    = {k: v.to(device) for k, v in inp.items()}
    logits = pace_model(**inp).logits
    probs  = torch.softmax(logits, dim=-1).cpu().numpy()

    results = []
    for i, p in enumerate(probs):
        pred  = int(np.argmax(p))
        label = pace_model.config.id2label[pred]
        results.append({
            "chunk_index"  : i + 1,
            "time_start_s" : i * CHUNK_SEC,
            "time_end_s"   : (i + 1) * CHUNK_SEC,
            "label"        : label,
            "stable_pct"   : round(float(p[0]) * 100, 2),
            "unstable_pct" : round(float(p[1]) * 100, 2),
        })
    return results


# ── Dynamic threshold helper ───────────────────────────────────────

def get_threshold(habit: str, chunk_count: int) -> float:
    """Return the dynamic threshold for a habit based on the number of chunks."""
    tiers = HABIT_THRESHOLDS.get(habit)
    if not tiers:
        return 0.50  # fallback
    for max_chunks, threshold in tiers:
        if max_chunks is None or chunk_count <= max_chunks:
            return threshold
    return tiers[-1][1]


def get_severity(rate: float, threshold: float) -> str:
    """Map a detection rate to a severity tier relative to its threshold."""
    if rate >= threshold * 2:
        return "high"
    if rate >= threshold:
        return "medium"
    return "low"


# ── Report builder ─────────────────────────────────────────────────

def build_report(audio_duration_s, filler_results, pace_results):
    n_filler   = sum(1 for r in filler_results if r["label"] == "Filler")
    n_unstable = sum(1 for r in pace_results   if r["label"].lower() == "unstable")
    total      = len(filler_results)

    filler_rate   = n_filler / max(total, 1)
    unstable_rate = n_unstable / max(total, 1)

    filler_thresh   = get_threshold("filler", total)
    pace_thresh     = get_threshold("pace", total)

    filler_flagged   = filler_rate >= filler_thresh
    pace_flagged     = unstable_rate >= pace_thresh

    filler_severity  = get_severity(filler_rate, filler_thresh)
    pace_severity    = get_severity(unstable_rate, pace_thresh)

    filler_verdict = (
        "HIGH FILLER"      if filler_severity == "high" else
        "MODERATE FILLER"  if filler_severity == "medium" else
        "CLEAN"
    )
    pace_verdict = (
        "HIGHLY UNSTABLE"    if pace_severity == "high" else
        "MODERATELY UNSTABLE" if pace_severity == "medium" else
        "STABLE"
    )

    return {
        "status"              : "success",
        "audio_duration_s"    : round(audio_duration_s, 2),
        "chunk_length_s"      : CHUNK_SEC,
        "total_chunks"        : total,

        "filler_detection": {
            "verdict"        : filler_verdict,
            "severity"       : filler_severity,
            "filler_chunks"  : n_filler,
            "clean_chunks"   : total - n_filler,
            "filler_rate_pct": round(100 * filler_rate, 1),
            "threshold_pct"  : round(100 * filler_thresh, 1),
            "avg_filler_confidence_pct": round(
                float(np.mean([r["filler_pct"] for r in filler_results])) if filler_results else 0, 2
            ),
            "flagged": filler_flagged,
            "chunks" : filler_results,
        },

        "pace_analysis": {
            "verdict"           : pace_verdict,
            "severity"          : pace_severity,
            "unstable_chunks"   : n_unstable,
            "stable_chunks"     : total - n_unstable,
            "instability_rate_pct": round(100 * unstable_rate, 1),
            "threshold_pct"     : round(100 * pace_thresh, 1),
            "avg_unstable_confidence_pct": round(
                float(np.mean([r["unstable_pct"] for r in pace_results])) if pace_results else 0, 2
            ),
            "flagged": pace_flagged,
            "chunks" : pace_results,
        },
    }


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy", "service": "bad-habit-detection"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Analyze an audio file for filler words and pace instability."""

    # Save uploaded file to a temp file for librosa
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp_path = None
    try:
        content = await file.read()

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE // (1024 * 1024)} MB.",
            )

        if file.content_type and file.content_type not in ALLOWED_AUDIO_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {file.content_type}. Allowed: {', '.join(ALLOWED_AUDIO_TYPES)}",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        audio      = load_audio(tmp_path)
        duration_s = len(audio) / SAMPLE_RATE
        chunks     = make_chunks(audio)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Audio is shorter than 5 seconds. Please provide a longer clip."
            )

        print(f"[INFO] Analyzing: {file.filename} | {duration_s:.1f}s | {len(chunks)} chunks")

        filler_results = run_filler(chunks)
        pace_results   = run_pace(chunks)

        report = build_report(duration_s, filler_results, pace_results)

        # Log output summary (without per-chunk detail to keep logs readable)
        summary = {
            "status"           : report["status"],
            "file_name"        : file.filename,
            "audio_duration_s" : report["audio_duration_s"],
            "total_chunks"     : report["total_chunks"],
            "filler_detection" : {
                "verdict"        : report["filler_detection"]["verdict"],
                "severity"       : report["filler_detection"]["severity"],
                "filler_rate_pct": report["filler_detection"]["filler_rate_pct"],
                "threshold_pct"  : report["filler_detection"]["threshold_pct"],
                "flagged"        : report["filler_detection"]["flagged"],
            },
            "pace_analysis"    : {
                "verdict"             : report["pace_analysis"]["verdict"],
                "severity"            : report["pace_analysis"]["severity"],
                "instability_rate_pct": report["pace_analysis"]["instability_rate_pct"],
                "threshold_pct"       : report["pace_analysis"]["threshold_pct"],
                "flagged"             : report["pace_analysis"]["flagged"],
            },
        }
        print(f"[RESULT] {json.dumps(summary, indent=2)}")

        return report

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Analysis failed for '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("analyze_audio:app", host="0.0.0.0", port=8001, reload=True)
