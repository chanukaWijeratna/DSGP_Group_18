import os
import torch
from torch import nn
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# --- Constants ---
SR = 16000
CHUNK_LENGTH = SR * 3  # 3 seconds per chunk

# ==========================================
# 1. MODEL ARCHITECTURES
# ==========================================


# NEW: Upgraded Clip-Level Stuttering Classifier
class StutteringClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec2.feature_extractor._freeze_parameters()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(hidden_size, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=512, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, num_classes)
        )

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        x = hidden_states.transpose(1, 2)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        logits = self.classifier(x)
        return logits


class SpeechClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec2.feature_extractor._freeze_parameters()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(hidden_size, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=512, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, num_classes)
        )

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        x = hidden_states.transpose(1, 2)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        return x


# ==========================================
# 2. SYSTEM INITIALIZATION
# ==========================================


def initialize_system(
    stutter_model_path="models/best_stutter_model.pth",
    slur_model_path="models/best_slurring_model.pth",
):
    print("Initializing Disorder Screening Models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    # Load NEW Stutter Model
    stutter_model = StutteringClassifier().to(device)
    if os.path.exists(stutter_model_path):
        stutter_model.load_state_dict(
            torch.load(stutter_model_path, map_location=device)
        )
        print("-> Stuttering model loaded successfully.")
    else:
        print(f"-> WARNING: Could not find {stutter_model_path}")
    stutter_model.eval()

    # Load Slur Model
    slur_model = SpeechClassifier().to(device)
    if os.path.exists(slur_model_path):
        slur_model.load_state_dict(torch.load(slur_model_path, map_location=device))
        print("-> Slurring model loaded successfully.")
    else:
        print(f"-> WARNING: Could not find {slur_model_path}")
    slur_model.eval()

    return processor, stutter_model, slur_model, device


# ==========================================
# 3. ANALYSIS PIPELINE (Performance Upgraded)
# ==========================================


MIN_CHUNK_LENGTH = SR * 1.5  # Discard chunks shorter than 1.5 seconds

# Dynamic stutter thresholds — scales with audio length (mirrors emotion system)
# Shorter clips need a higher % of flagged chunks to avoid false positives;
# longer clips use a lower bar because even a small fraction is significant.
# Format: list of (max_chunk_count, threshold_percent) checked in order.
STUTTER_THRESHOLDS = [
    (5,   30.0),   # ≤15 s  → need 30% of chunks flagged
    (20,  20.0),   # ≤1 min → 20%
    (40,  15.0),   # ≤2 min → 15%
    (None, 10.0),  # >2 min → 10%
]

SLUR_THRESHOLD = 0.6         # Flag slurring only if > 60% of chunks are dysarthric
MIN_CHUNK_CONFIDENCE = 0.65  # Ignore chunk predictions below 65% confidence


def get_stutter_threshold(chunk_count: int) -> float:
    """Return the dynamic stutter severity threshold based on audio length."""
    for max_chunks, threshold in STUTTER_THRESHOLDS:
        if max_chunks is None or chunk_count <= max_chunks:
            return threshold
    return STUTTER_THRESHOLDS[-1][1]


def analyze_audio(audio_path, processor, stutter_model, slur_model, device):
    """Processes the audio file through both models with confidence scoring."""
    waveform, _ = librosa.load(audio_path, sr=SR)

    slur_predictions = []
    stutter_predictions = []
    slur_confidences = []
    stutter_confidences = []

    # Chop audio into 3-second blocks
    raw_chunks = [
        waveform[i : i + CHUNK_LENGTH] for i in range(0, len(waveform), CHUNK_LENGTH)
    ]

    # Drop last chunk if it's too short to be reliable
    chunks = [c for c in raw_chunks if len(c) >= MIN_CHUNK_LENGTH]

    if len(chunks) == 0:
        return {
            "stuttering": {"flagged_chunks": [], "severity_percentage": 0, "confidence_avg": 0, "flagged": False},
            "slurring": {"flagged_chunks": [], "fluent_chunks": 0, "dysarthric_ratio": 0, "confidence_avg": 0, "flagged": False},
        }

    with torch.no_grad():
        for chunk in chunks:
            # Pad chunks that are slightly short (between 1.5s and 3s)
            if len(chunk) < CHUNK_LENGTH:
                chunk = np.pad(chunk, (0, CHUNK_LENGTH - len(chunk)))

            inputs = processor(
                chunk,
                sampling_rate=SR,
                return_tensors="pt",
                padding="max_length",
                max_length=CHUNK_LENGTH,
            )
            input_values = inputs.input_values.to(device)

            # --- SLURRING INFERENCE with confidence ---
            slur_outputs = slur_model(input_values)
            slur_probs = torch.softmax(slur_outputs, dim=1)
            slur_pred = torch.argmax(slur_probs, dim=1).item()
            slur_conf = slur_probs[0, slur_pred].item()
            # Only count positive predictions that meet confidence threshold
            if slur_pred == 1 and slur_conf < MIN_CHUNK_CONFIDENCE:
                slur_pred = 0  # treat low-confidence positive as negative
            slur_predictions.append(slur_pred)
            slur_confidences.append(slur_conf)

            # --- STUTTERING INFERENCE with confidence ---
            stutter_outputs = stutter_model(input_values)
            stutter_probs = torch.softmax(stutter_outputs, dim=1)
            stutter_pred = torch.argmax(stutter_probs, dim=1).item()
            stutter_conf = stutter_probs[0, stutter_pred].item()
            print(f"  [STUTTER DEBUG] Chunk raw logits: {stutter_outputs.cpu().numpy()}, "
                  f"probs: [class0={stutter_probs[0,0].item():.4f}, class1={stutter_probs[0,1].item():.4f}], "
                  f"pred={stutter_pred}, conf={stutter_conf:.4f}")
            # No confidence gating — trust the model's prediction
            stutter_predictions.append(stutter_pred)
            stutter_confidences.append(stutter_conf)

    total_chunks = len(chunks)

    # --- AGGREGATE SLURRING RESULTS ---
    flagged_slur_chunks = [i for i, pred in enumerate(slur_predictions) if pred == 1]
    dysarthric_count = len(flagged_slur_chunks)
    control_count = total_chunks - dysarthric_count
    slur_ratio = dysarthric_count / total_chunks
    is_slurred = bool(slur_ratio > SLUR_THRESHOLD)

    # --- AGGREGATE STUTTERING RESULTS ---
    flagged_stutter_chunks = [i for i, pred in enumerate(stutter_predictions) if pred == 1]
    stutter_chunks_count = len(flagged_stutter_chunks)
    stutter_severity_percent = (stutter_chunks_count / total_chunks) * 100
    is_stutter_flagged = bool(stutter_severity_percent > get_stutter_threshold(total_chunks))

    results = {
        "stuttering": {
            "flagged_chunks": flagged_stutter_chunks,
            "severity_percentage": round(stutter_severity_percent, 2),
            "confidence_avg": round(sum(stutter_confidences) / len(stutter_confidences) * 100, 2),
            "flagged": is_stutter_flagged,
        },
        "slurring": {
            "flagged_chunks": flagged_slur_chunks,
            "fluent_chunks": control_count,
            "dysarthric_ratio": round(slur_ratio * 100, 2),
            "confidence_avg": round(sum(slur_confidences) / len(slur_confidences) * 100, 2),
            "flagged": is_slurred,
        },
        "total_chunks_analyzed": total_chunks,
    }

    return results
