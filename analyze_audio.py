# Bad Habit Detection API — Filler Words + Pace Instability
# Run: uvicorn analyze_audio:app --host 0.0.0.0 --port 8001

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

# Config
SAMPLE_RATE = 16_000

# Filler: 1s window, 0.3s hop
FILLER_CLIP_S       = 1.0
FILLER_CLIP_SAMPLES = int(SAMPLE_RATE * FILLER_CLIP_S)
FILLER_HOP_S        = 0.3
DEFAULT_FILLER_THRESH = 0.50

# Pace: 5s chunks, no overlap
PACE_CHUNK_S       = 5
PACE_CHUNK_SAMPLES = SAMPLE_RATE * PACE_CHUNK_S

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3",
    "audio/ogg", "audio/webm", "audio/flac", "audio/mp4",
}

# Thresholds scale with audio length- longer clips use lower %
HABIT_THRESHOLDS = {
    "filler": [
        (10,   0.50),
        (40,   0.35),
        (80,   0.25),
        (None, 0.15),
    ],
    "pace": [
        (3,    0.55),
        (12,   0.40),
        (24,   0.30),
        (None, 0.20),
    ],
}

# Model paths
FILLER_MODEL_DIR = os.path.join(os.path.dirname(__file__), "filler_detector")
PACE_MODEL_DIR   = os.path.join(os.path.dirname(__file__), "pace_model")


# App
app = FastAPI(
    title="Bad Habit Detection API",
    description="Filler word detection and pace instability analysis.",
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


# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

# Filler model
print(f"[INFO] Loading filler model from '{FILLER_MODEL_DIR}' ...")
filler_fe    = AutoFeatureExtractor.from_pretrained(FILLER_MODEL_DIR)
filler_model = AutoModelForAudioClassification.from_pretrained(FILLER_MODEL_DIR).to(device)
filler_model.eval()

# grab the optimal threshold from training if it exists
config_extra_path = os.path.join(FILLER_MODEL_DIR, "config_extra.json")
if os.path.exists(config_extra_path):
    with open(config_extra_path) as f:
        _cfg = json.load(f)
    OPT_FILLER_THRESH = float(_cfg.get("optimal_threshold", DEFAULT_FILLER_THRESH))
    print(f"[INFO] Filler threshold: {OPT_FILLER_THRESH:.2f}")
else:
    OPT_FILLER_THRESH = DEFAULT_FILLER_THRESH
    print(f"[WARN] config_extra.json missing, using default threshold {DEFAULT_FILLER_THRESH}")
print("[INFO] Filler model ready.")

# Pace model
print(f"[INFO] Loading pace model from '{PACE_MODEL_DIR}' ...")
pace_fe    = Wav2Vec2FeatureExtractor.from_pretrained(PACE_MODEL_DIR)
pace_model = Wav2Vec2ForSequenceClassification.from_pretrained(PACE_MODEL_DIR).to(device)
pace_model.eval()
print("[INFO] Pace model ready.")


# Audio helpers

def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio


def make_filler_windows(audio: np.ndarray):
    """1s window, 0.3s hop — same as notebook scan_audio()"""
    ws = int(FILLER_CLIP_S * SAMPLE_RATE)
    hs = int(FILLER_HOP_S * SAMPLE_RATE)
    windows = []
    for s in range(0, len(audio) - ws + 1, hs):
        clip = audio[s : s + ws]
        pk = np.abs(clip).max()
        if pk > 0:
            clip = clip / pk
        windows.append((s, clip))
    return windows


def make_pace_chunks(audio: np.ndarray):
    """5s non-overlapping chunks"""
    return [
        audio[s : s + PACE_CHUNK_SAMPLES]
        for s in range(0, len(audio) - PACE_CHUNK_SAMPLES + 1, PACE_CHUNK_SAMPLES)
    ]


# Inference

@torch.no_grad()
def run_filler(windows):
    clips = [clip for _, clip in windows]

    inp = filler_fe(
        clips,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding="max_length",
        max_length=FILLER_CLIP_SAMPLES,
        truncation=True,
    )
    inp = {k: v.to(device) for k, v in inp.items()}
    logits = filler_model(**inp).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    results = []
    for i, (start_sample, _) in enumerate(windows):
        p = probs[i]
        pred = 1 if float(p[1]) >= OPT_FILLER_THRESH else 0
        label = filler_model.config.id2label[pred]
        start_s = start_sample / SAMPLE_RATE
        end_s = start_s + FILLER_CLIP_S

        results.append({
            "window_index": i + 1,
            "time_start_s": round(start_s, 2),
            "time_end_s": round(end_s, 2),
            "label": label,
            "confidence": round(float(p[pred]) * 100, 2),
            "filler_pct": round(float(p[1]) * 100, 2),
            "clean_pct": round(float(p[0]) * 100, 2),
        })
    return results


@torch.no_grad()
def run_pace(chunks):
    inp = pace_fe(
        chunks,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        max_length=PACE_CHUNK_SAMPLES,
        truncation=True,
    )
    inp = {k: v.to(device) for k, v in inp.items()}
    logits = pace_model(**inp).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    results = []
    for i, p in enumerate(probs):
        pred = int(np.argmax(p))
        label = pace_model.config.id2label[pred]
        results.append({
            "chunk_index": i + 1,
            "time_start_s": i * PACE_CHUNK_S,
            "time_end_s": (i + 1) * PACE_CHUNK_S,
            "label": label,
            "stable_pct": round(float(p[0]) * 100, 2),
            "unstable_pct": round(float(p[1]) * 100, 2),
        })
    return results


# Threshold helpers

def get_threshold(habit: str, count: int) -> float:
    tiers = HABIT_THRESHOLDS.get(habit)
    if not tiers:
        return 0.50
    for max_count, threshold in tiers:
        if max_count is None or count <= max_count:
            return threshold
    return tiers[-1][1]


def get_severity(rate: float, threshold: float) -> str:
    if rate >= threshold * 2:
        return "high"
    if rate >= threshold:
        return "medium"
    return "low"


# Build report

def build_report(audio_duration_s, filler_results, pace_results):
    # filler stats
    n_filler_wins = len(filler_results)
    n_filler = sum(1 for r in filler_results if r["label"] == "Filler")
    filler_rate = n_filler / max(n_filler_wins, 1)
    filler_thresh = get_threshold("filler", n_filler_wins)
    filler_flagged = filler_rate >= filler_thresh
    filler_severity = get_severity(filler_rate, filler_thresh)

    filler_verdict = (
        "HIGH FILLER" if filler_severity == "high" else
        "MODERATE FILLER" if filler_severity == "medium" else
        "CLEAN"
    )

    # pace stats
    n_pace_chunks = len(pace_results)
    n_unstable = sum(1 for r in pace_results if r["label"].lower() == "unstable")
    unstable_rate = n_unstable / max(n_pace_chunks, 1)
    pace_thresh = get_threshold("pace", n_pace_chunks)
    pace_flagged = unstable_rate >= pace_thresh
    pace_severity = get_severity(unstable_rate, pace_thresh)

    pace_verdict = (
        "HIGHLY UNSTABLE" if pace_severity == "high" else
        "MODERATELY UNSTABLE" if pace_severity == "medium" else
        "STABLE"
    )

    return {
        "status": "success",
        "audio_duration_s": round(audio_duration_s, 2),

        "filler_detection": {
            "verdict": filler_verdict,
            "severity": filler_severity,
            "window_length_s": FILLER_CLIP_S,
            "window_hop_s": FILLER_HOP_S,
            "total_windows": n_filler_wins,
            "filler_windows": n_filler,
            "clean_windows": n_filler_wins - n_filler,
            "filler_rate_pct": round(100 * filler_rate, 1),
            "threshold_pct": round(100 * filler_thresh, 1),
            "optimal_threshold": OPT_FILLER_THRESH,
            "avg_filler_confidence_pct": round(
                float(np.mean([r["filler_pct"] for r in filler_results]))
                if filler_results else 0, 2
            ),
            "flagged": filler_flagged,
            "windows": filler_results,
        },

        "pace_analysis": {
            "verdict": pace_verdict,
            "severity": pace_severity,
            "chunk_length_s": PACE_CHUNK_S,
            "total_chunks": n_pace_chunks,
            "unstable_chunks": n_unstable,
            "stable_chunks": n_pace_chunks - n_unstable,
            "instability_rate_pct": round(100 * unstable_rate, 1),
            "threshold_pct": round(100 * pace_thresh, 1),
            "avg_unstable_confidence_pct": round(
                float(np.mean([r["unstable_pct"] for r in pace_results]))
                if pace_results else 0, 2
            ),
            "flagged": pace_flagged,
            "chunks": pace_results,
        },
    }


# Endpoints

@app.get("/health")
def health():
    return {"status": "healthy", "service": "bad-habit-detection"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp_path = None

    try:
        content = await file.read()

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max {MAX_FILE_SIZE // (1024 * 1024)} MB.",
            )

        if file.content_type and file.content_type not in ALLOWED_AUDIO_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {file.content_type}. "
                       f"Allowed: {', '.join(ALLOWED_AUDIO_TYPES)}",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        audio = load_audio(tmp_path)
        duration_s = len(audio) / SAMPLE_RATE

        # run both
        filler_windows = make_filler_windows(audio)
        pace_chunks = make_pace_chunks(audio)

        if not filler_windows and not pace_chunks:
            raise HTTPException(
                status_code=400,
                detail="Audio too short. Need at least 1s for filler or 5s for pace.",
            )

        print(f"[INFO] {file.filename} | {duration_s:.1f}s | "
              f"{len(filler_windows)} filler windows | {len(pace_chunks)} pace chunks")

        filler_results = run_filler(filler_windows) if filler_windows else []
        pace_results = run_pace(pace_chunks) if pace_chunks else []

        report = build_report(duration_s, filler_results, pace_results)

        # log
        summary = {
            "status": report["status"],
            "file": file.filename,
            "duration_s": report["audio_duration_s"],
            "filler": {
                "verdict": report["filler_detection"]["verdict"],
                "severity": report["filler_detection"]["severity"],
                "rate": report["filler_detection"]["filler_rate_pct"],
                "flagged": report["filler_detection"]["flagged"],
            },
            "pace": {
                "verdict": report["pace_analysis"]["verdict"],
                "severity": report["pace_analysis"]["severity"],
                "rate": report["pace_analysis"]["instability_rate_pct"],
                "flagged": report["pace_analysis"]["flagged"],
            },
        }
        print(f"[RESULT] {json.dumps(summary, indent=2)}")

        return report

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Failed for '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# Main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("analyze_audio:app", host="0.0.0.0", port=8001, reload=True)
