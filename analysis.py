# analyze_features.py
"""
Generate summary stats and figures from target_features.csv.

Usage:
  python analyze_features.py --in_csv target_features.csv --out_dir reports/figures
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_hist(df: pd.DataFrame, col: str, out_dir: str, bins: int = 20):
    if col not in df.columns:
        return
    data = df[col].dropna().to_numpy()
    if data.size == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"hist_{col}.png"), dpi=150)
    plt.close()


def save_boxplot(df: pd.DataFrame, cols: list, out_dir: str, title: str, fname: str):
    avail = [c for c in cols if c in df.columns]
    if not avail:
        return
    plt.figure(figsize=(8, 5))
    df[avail].boxplot()
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()


def save_corr_heatmap(df: pd.DataFrame, out_dir: str, fname: str = "corr_heatmap.png"):
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        return
    corr = num_df.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(corr.shape[1])
    labels = list(corr.columns)
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    plt.title("Correlation between morphological features")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", default="target_features.csv", help="Path to features CSV")
    parser.add_argument("--out_dir", default="reports/figures", help="Directory to save figures and tables")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # Load data
    df = pd.read_csv(args.in_csv)

    # Save summary stats
    # Patch for older pandas versions
    summary = df.select_dtypes(include=[np.number]).describe().T

    summary_path = os.path.join(args.out_dir, "summary_stats.csv")
    summary.to_csv(summary_path, index=True)

    # Key features (only plot those that exist)
    key_feats = [
        "total_length_um",
        "num_segments",
        "n_bifurcations",
        "n_tips",
        "max_tree_depth",
        "sholl_peak",
        "sholl_auc",
        "max_radial_extent_um",
        "mean_segment_length_um",
        "std_segment_length_um",
        "bif_per_100um",
        "tips_per_100um",
    ]

    # Histograms
    for col in key_feats:
        save_hist(df, col, args.out_dir, bins=20)

    # Boxplots
    save_boxplot(
        df,
        ["total_length_um", "num_segments", "n_bifurcations", "n_tips"],
        args.out_dir,
        title="Boxplots: branching & size features",
        fname="boxplot_core.png",
    )

    # Correlation heatmap
    save_corr_heatmap(df, args.out_dir, fname="corr_heatmap.png")

    # Also save a quick preview table
    head_path = os.path.join(args.out_dir, "head_preview.csv")
    df.head(10).to_csv(head_path, index=False)

    print(f"[ok] Wrote: {summary_path}")
    print(f"[ok] Figures saved under: {args.out_dir}")
    print(f"[ok] Preview rows: {head_path}")


if __name__ == "__main__":
    main()
