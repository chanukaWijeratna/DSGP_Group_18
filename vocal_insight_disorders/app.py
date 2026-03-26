"""
====================================================================
  Disorder Detection API — Stuttering & Slurring Detection
  Component   : Speech Disorder Analysis
  Framework   : FastAPI
  Port        : 5000
====================================================================

TO RUN (from backend/disorder-detection/routes/):
    uvicorn app:app --host 0.0.0.0 --port 5000

ENDPOINTS:
    POST /api/analyze/disorder   — Accept audio file, return disorder analysis
    GET  /health                 — Health check
====================================================================
"""

import os
import uuid
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from disorder_service import initialize_system, analyze_audio

# ── FastAPI app ────────────────────────────────────────────────────
app = FastAPI(
    title="Disorder Detection API",
    description="Stuttering and slurring detection for Vocal Insight.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize models once on startup ─────────────────────────────
print("Starting server and loading models...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
processor, stutter_model, slur_model, device = initialize_system(
    stutter_model_path=os.path.join(BASE_DIR, "../models/best_stutter_model.pth"),
    slur_model_path=os.path.join(BASE_DIR, "../models/best_slurring_model.pth"),
)


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy", "service": "disorder-detection"}


@app.post("/api/analyze/disorder")
async def analyze_disorder(audio: UploadFile = File(...)):
    """Analyze an audio file for stuttering and slurring."""

    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    filepath = None

    try:
        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            filepath = tmp.name

        # Run disorder analysis
        results = analyze_audio(
            filepath, processor, stutter_model, slur_model, device
        )

        return {
            "success": True,
            "message": "Speech disorder analysis complete",
            "data": results,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during analysis: {str(e)}",
        )

    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
