import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sentence_transformers import SentenceTransformer

# ── re-use data structures from retrieve.py ──────────────────────────────────
from retrieve import configs, test_queries, load_config_data, retrieve

REMARKS = {
    # MiniLM, chunk=256, threshold=0.3
    "R1":  "MiniLM c256 k=1 single best",
    "R2":  "MiniLM c256 k=3 baseline",
    "R3":  "MiniLM c256 k=5 moderate",
    "R4":  "MiniLM c256 k=7 wider recall",
    "R5":  "MiniLM c256 k=10 max recall",
    # MiniLM, chunk=512, threshold=0.3
    "R6":  "MiniLM c512 k=1 single best",
    "R7":  "MiniLM c512 k=3 baseline",
    "R8":  "MiniLM c512 k=5 moderate",
    "R9":  "MiniLM c512 k=7 wider recall",
    "R10": "MiniLM c512 k=10 max recall",
    # mpnet, chunk=256, threshold=0.5
    "R11": "mpnet c256 k=1 single best",
    "R12": "mpnet c256 k=3 baseline",
    "R13": "mpnet c256 k=5 moderate",
    "R14": "mpnet c256 k=7 wider recall",
    "R15": "mpnet c256 k=10 max recall",
    # mpnet, chunk=512, threshold=0.5
    "R16": "mpnet c512 k=1 single best",
    "R17": "mpnet c512 k=3 baseline",
    "R18": "mpnet c512 k=5 moderate",
    "R19": "mpnet c512 k=7 wider recall",
    "R20": "mpnet c512 k=10 max recall",
}


# ── metrics ──────────────────────────────────────────────────────────────────

def is_relevant(result: dict, expected_files: list) -> bool:
    return result["source"] in expected_files


def compute_metrics(results: list, expected_files: list):
    """
    results        : list of retrieved chunk dicts (may be empty)
    expected_files : list of expected source filenames

    Returns (precision, recall, reciprocal_rank).
    """
    if not results:
        return 0.0, 0.0, 0.0

    relevant_hits = [is_relevant(r, expected_files) for r in results]

    precision = sum(relevant_hits) / len(results)
    recall = 1.0 if any(relevant_hits) else 0.0

    rr = 0.0
    for rank, hit in enumerate(relevant_hits, start=1):
        if hit:
            rr = 1.0 / rank
            break

    return precision, recall, rr


# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate():
    # Pre-load models and embeddings (shared across configs with same key)
    models = {}
    config_data = {}

    for cfg in configs:
        model_name = cfg["embedding"]
        data_key = f"{model_name}_chunk{cfg['chunk_size']}"
        if model_name not in models:
            print(f"  Loading model: {model_name}")
            models[model_name] = SentenceTransformer(model_name)
        if data_key not in config_data:
            print(f"  Loading embeddings: {data_key}")
            config_data[data_key] = load_config_data(model_name, cfg["chunk_size"])

    rows = []
    for cfg in configs:
        model_name = cfg["embedding"]
        data_key = f"{model_name}_chunk{cfg['chunk_size']}"
        embeddings, chunks = config_data[data_key]
        model = models[model_name]

        precisions, recalls, rrs = [], [], []

        for tq in test_queries:
            results = retrieve(tq["query"], cfg, model, embeddings, chunks)
            p, r, rr = compute_metrics(results, tq["expected_files"])
            precisions.append(p)
            recalls.append(r)
            rrs.append(rr)

        avg_p   = round(np.mean(precisions), 4)
        avg_r   = round(np.mean(recalls),    4)
        avg_mrr = round(np.mean(rrs),        4)

        print(f"  {cfg['name']}: Precision={avg_p:.4f}  Recall={avg_r:.4f}  MRR={avg_mrr:.4f}")

        rows.append({
            "Config":               cfg["name"],
            "Embedding Model":      cfg["embedding"],
            "Chunk Size (tokens)":  cfg["chunk_size"],
            "Top-K":                cfg["top_k"],
            "Similarity Threshold": cfg["threshold"],
            "Retrieval Precision":  avg_p,
            "Retrieval Recall":     avg_r,
            "MRR":                  avg_mrr,
            "Remark":               REMARKS[cfg["name"]],
        })

    return pd.DataFrame(rows)


# ── report rendering ──────────────────────────────────────────────────────────

def render_table(df: pd.DataFrame, output_path: str = "retrieval_report.png"):
    col_order = [
        "Config", "Embedding Model", "Chunk Size (tokens)", "Top-K",
        "Similarity Threshold", "Retrieval Precision", "Retrieval Recall",
        "MRR", "Remark",
    ]
    df = df[col_order]

    # Format numeric metric columns to 4 dp
    for col in ["Retrieval Precision", "Retrieval Recall", "MRR"]:
        df[col] = df[col].apply(lambda x: f"{x:.4f}")

    col_widths = [0.06, 0.14, 0.12, 0.07, 0.12, 0.13, 0.12, 0.08, 0.18]

    fig_w = 18
    row_h = 0.55
    header_h = 0.9
    fig_h = header_h + row_h * len(df) + 0.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    BG       = "#1a1a2e"
    HDR_BG   = "#16213e"
    HDR_FG   = "#e0e0e0"
    ROW_ODD  = "#1a1a2e"
    ROW_EVEN = "#162032"
    FG       = "#d0d0d0"
    ACCENT   = "#c0392b"
    BORDER   = "#2c3e50"

    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    n_rows, n_cols = len(df), len(col_order)

    # cumulative x positions
    x_positions = [0.0]
    for w in col_widths:
        x_positions.append(x_positions[-1] + w)

    total_w = x_positions[-1]
    total_h = header_h + row_h * n_rows

    # ── draw cells ────────────────────────────────────────────────────────────
    def draw_cell(row_idx, col_idx, text, bg, fg, bold=False, wrap=False):
        x0 = x_positions[col_idx]
        w  = col_widths[col_idx]
        if row_idx == -1:          # header
            y0 = total_h - header_h
            h  = header_h
        else:
            y0 = total_h - header_h - (row_idx + 1) * row_h
            h  = row_h

        rect = mpatches.FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="square,pad=0",
            linewidth=0.5,
            edgecolor=BORDER,
            facecolor=bg,
            transform=ax.transData,
            clip_on=False,
        )
        ax.add_patch(rect)

        fontsize = 8.5 if not wrap else 7.8
        weight   = "bold" if bold else "normal"
        ax.text(
            x0 + w / 2,
            y0 + h / 2,
            text,
            ha="center", va="center",
            fontsize=fontsize,
            fontweight=weight,
            color=fg,
            wrap=True,
            transform=ax.transData,
            clip_on=False,
        )

    # header
    for ci, col in enumerate(col_order):
        draw_cell(-1, ci, col, HDR_BG, HDR_FG, bold=True, wrap=True)

    # data rows
    for ri, (_, row) in enumerate(df.iterrows()):
        bg = ROW_ODD if ri % 2 == 0 else ROW_EVEN
        for ci, col in enumerate(col_order):
            val = str(row[col])
            # highlight config name
            fg_col = ACCENT if col == "Config" else FG
            draw_cell(ri, ci, val, bg, fg_col, bold=(col == "Config"), wrap=(col == "Remark"))

    # outer border
    outer = mpatches.FancyBboxPatch(
        (0, 0), total_w, total_h,
        boxstyle="square,pad=0",
        linewidth=1.5,
        edgecolor=ACCENT,
        facecolor="none",
        transform=ax.transData,
        clip_on=False,
    )
    ax.add_patch(outer)

    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)

    plt.title("Retrieval Configuration Evaluation Report",
              fontsize=13, fontweight="bold", color=HDR_FG, pad=10,
              backgroundcolor=BG)

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nReport saved: {output_path}")
    plt.show()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running retrieval evaluation across all configs...\n")
    df = evaluate()

    print("\n-- Summary --------------------------------------------------")
    print(df[["Config", "Retrieval Precision", "Retrieval Recall", "MRR"]].to_string(index=False))

    # Save CSV
    df.to_csv("retrieval_report.csv", index=False)
    print("\nCSV saved: retrieval_report.csv")

    # Render table image
    render_table(df, "retrieval_report.png")
