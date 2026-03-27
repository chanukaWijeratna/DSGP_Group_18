"""
====================================================================
  Feedback System API — Multi-modal Speech Analysis & Feedback BOT
  Component   : RAG Feedback Generation
  Framework   : FastAPI
  Port        : 5001
====================================================================

TO RUN (from backend/feedback-system/):
    uvicorn app:app --host 0.0.0.0 --port 5001

ENDPOINTS:
    POST /feedback   — Accept emotion + disorder JSON, run RAG, save session
    GET  /results/{user_id}                    — List all results for a user
    GET  /results/{user_id}/{session_id}       — Retrieve a specific result
    GET  /health     — Health check
====================================================================
"""

import os
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import generate_feedback

# ── Sessions directory ──────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── FastAPI app ─────────────────────────────────────────────────
app = FastAPI(
    title="Feedback System API",
    description="RAG-based speech feedback generation for Vocal Insight (Group 18).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request schema ───────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    user_id: str
    session_id: str
    scenario: Optional[str] = ""
    emotion_result: Optional[Dict[str, Any]] = None
    disorder_result: Optional[Dict[str, Any]] = None
    bad_habit_result: Optional[Dict[str, Any]] = None


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy", "service": "feedback-system"}


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """
    Run the RAG feedback pipeline on emotion + disorder model outputs,
    embed feedback into the original JSONs, and save the full session
    to sessions/{user_id}/{session_id}.json.
    """
    result = generate_feedback(
        emotion_result=req.emotion_result,
        disorder_result=req.disorder_result,
        bad_habit_result=req.bad_habit_result,
        scenario=req.scenario,
    )

    # Derive detected problem names for the summary list
    problems_detected = []
    dr = result.get("disorder_result") or {}
    if dr.get("stuttering", {}).get("flagged"):
        problems_detected.append("stuttering")
    if dr.get("slurring", {}).get("flagged"):
        problems_detected.append("slurring")
    er = result.get("emotion_result") or {}
    problems_detected.extend((er.get("rag_feedback") or {}).keys())
    bh = result.get("bad_habit_result") or {}
    if bh.get("filler_detection", {}).get("flagged"):
        problems_detected.append("filler_words")
    if bh.get("pace_analysis", {}).get("flagged"):
        problems_detected.append("fast_speaking_rate")

    session_data = {
        "user_id":           req.user_id,
        "session_id":        req.session_id,
        "scenario":          req.scenario,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "problems_detected": problems_detected,
        "emotion_result":    result.get("emotion_result"),
        "disorder_result":   result.get("disorder_result"),
        "bad_habit_result":  result.get("bad_habit_result"),
    }

    # Only save when at least one problem was detected
    if problems_detected:
        user_dir = os.path.join(RESULTS_DIR, req.user_id)
        os.makedirs(user_dir, exist_ok=True)

        filepath = os.path.join(user_dir, f"{req.session_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Result saved: {filepath}")
    else:
        print(f"○ No problems detected — skipping save for session {req.session_id}")

    return session_data


@app.get("/results/{user_id}")
def list_results(user_id: str):
    """Return a list of all result files for a user."""
    user_dir = os.path.join(RESULTS_DIR, user_id)

    if not os.path.exists(user_dir):
        return {"user_id": user_id, "results": []}

    results = []
    for fname in sorted(os.listdir(user_dir), reverse=True):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(user_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append({
                "session_id":        data.get("session_id"),
                "timestamp":         data.get("timestamp"),
                "scenario":          data.get("scenario"),
                "problems_detected": data.get("problems_detected", []),
            })
        except Exception:
            pass

    return {"user_id": user_id, "results": results}


@app.get("/results/{user_id}/severity-history")
def severity_history(user_id: str):
    """Return severity trend data across all sessions for charting."""
    user_dir = os.path.join(RESULTS_DIR, user_id)

    if not os.path.exists(user_dir):
        return {"user_id": user_id, "history": []}

    history = []
    for fname in sorted(os.listdir(user_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(user_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            severity = {}
            dr = data.get("disorder_result") or {}
            st = dr.get("stuttering") or {}
            if "severity_percentage" in st:
                severity["stuttering"] = st["severity_percentage"]
            sl = dr.get("slurring") or {}
            if "dysarthric_ratio" in sl:
                severity["slurring"] = sl["dysarthric_ratio"]

            bh = data.get("bad_habit_result") or {}
            fd = bh.get("filler_detection") or {}
            if "filler_rate_pct" in fd:
                severity["filler_words"] = fd["filler_rate_pct"]
            pa = bh.get("pace_analysis") or {}
            if "instability_rate_pct" in pa:
                severity["fast_speaking_rate"] = pa["instability_rate_pct"]

            history.append({
                "session_id": data.get("session_id"),
                "timestamp":  data.get("timestamp"),
                "severity":   severity,
            })
        except Exception:
            pass

    return {"user_id": user_id, "history": history}


@app.get("/results/{user_id}/{session_id}")
def get_result(user_id: str, session_id: str):
    """Return the full saved JSON for a specific result."""
    filepath = os.path.join(RESULTS_DIR, user_id, f"{session_id}.json")

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Session not found")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


@app.delete("/results/{user_id}/{session_id}")
def delete_result(user_id: str, session_id: str):
    """Permanently delete a saved session result."""
    filepath = os.path.join(RESULTS_DIR, user_id, f"{session_id}.json")

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Session not found")

    os.remove(filepath)
    return {"deleted": True, "session_id": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)
