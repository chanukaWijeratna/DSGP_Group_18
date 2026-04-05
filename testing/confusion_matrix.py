import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sentence_transformers import SentenceTransformer

from retrieve import configs, test_queries, load_config_data, retrieve

# ── Pick config ──
cfg = next(c for c in configs if c["name"] == "R7")

print(f"Config: {cfg}")
print(f"Total test queries: {len(test_queries)}\n")

# ── Load model & embeddings ──
model = SentenceTransformer(cfg["embedding"])
embeddings, chunks = load_config_data(cfg["embedding"], cfg["chunk_size"])

# ── Run retrieval on every query ──
y_true = []   # expected file label
y_pred = []   # predicted file label (top-1 source)

for i, tq in enumerate(test_queries):
    results = retrieve(tq["query"], cfg, model, embeddings, chunks)
    expected = tq["expected_files"][0]  # single expected file

    if results:
        predicted = results[0]["source"]  # top-1 result
    else:
        predicted = "NO_RESULT"

    y_true.append(expected)
    y_pred.append(predicted)

    match = "OK" if predicted == expected else "MISS"
    print(f"  Q{i+1:02d} [{match}]  expected={expected:<28s}  predicted={predicted:<28s}")

# ── Build confusion matrix ──
labels = sorted(set(y_true + y_pred))
# Shorter display labels
short = {l: l.replace(".md", "").replace("_", " ").title() for l in labels}
display_labels = [short[l] for l in labels]

cm = confusion_matrix(y_true, y_pred, labels=labels)

print("\n" + "=" * 60)
print("Classification Report (treating top-1 retrieval as prediction):")
print("=" * 60)
print(classification_report(y_true, y_pred, labels=labels, target_names=display_labels, zero_division=0))

# ── Plot ──
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
ax.set_title("Confusion Matrix — Top-1 Retrieved File vs Expected File", fontsize=12, fontweight="bold")
ax.set_xlabel("Predicted (Top-1 Retrieved File)", fontsize=10)
ax.set_ylabel("Actual (Expected File)", fontsize=10)
plt.xticks(rotation=35, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
print("\nSaved: confusion_matrix.png")
plt.show()
