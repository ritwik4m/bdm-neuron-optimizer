#!/usr/bin/env python3
import argparse, json, os, glob
import pandas as pd
import numpy as np
from swc_metrics import compute_features_from_swc

def plot_swc_2d(swc_file, out_png):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(
        swc_file, comment="#", sep=r"\s+", header=None,
        names=["id", "type", "x", "y", "z", "radius", "parent"], engine="python"
    )
    # Simple 2D projection (x–y), draw edges child→parent
    seg = df[df["parent"] != -1].copy()
    parents = df.set_index("id")[["x","y"]].rename(columns={"x":"x_p","y":"y_p"})
    seg = seg.join(parents, on="parent", how="left")
    plt.figure(figsize=(6,6))
    for _, r in seg.dropna(subset=["x_p","y_p"]).iterrows():
        plt.plot([r["x_p"], r["x"]], [r["y_p"], r["y"]], linewidth=0.8)
    soma = df[df["type"]==1][["x","y"]]
    if not soma.empty:
        plt.scatter(soma["x"], soma["y"], s=20)
    plt.axis("equal"); plt.xlabel("x (µm)"); plt.ylabel("y (µm)")
    plt.title(os.path.basename(swc_file))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    p = argparse.ArgumentParser(description="Batch SWC metrics → CSV + summary + example figure")
    p.add_argument("--in_dir", required=True, help="Folder with .swc files")
    p.add_argument("--out_csv", required=True, help="Output CSV for per-neuron metrics")
    p.add_argument("--summary_csv", default=None, help="Output CSV for mean/std (default: alongside out_csv)")
    p.add_argument("--example_png", default=None, help="Output PNG visualizing one representative SWC")
    p.add_argument("--pick", default="median", choices=["median","longest","shortest"],
                   help="How to pick the example SWC (default: median total_length_um)")
    p.add_argument("--recursive", action="store_true", help="Search subfolders recursively")
    args = p.parse_args()

    pattern = "**/*.swc" if args.recursive else "*.swc"
    files = sorted(glob.glob(os.path.join(args.in_dir, pattern), recursive=args.recursive))
    if not files:
        raise SystemExit(f"No SWC files found in: {args.in_dir}")

    rows = []
    for f in files:
        feats = compute_features_from_swc(f)
        feats["file"] = os.path.basename(f)
        feats["path"] = os.path.abspath(f)
        rows.append(feats)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # Summary (mean/std across numeric columns)
    num_cols = df.select_dtypes(include=[np.number]).columns
    summary = pd.DataFrame({
        "mean": df[num_cols].mean(numeric_only=True),
        "std": df[num_cols].std(numeric_only=True, ddof=1)
    })
    if args.summary_csv is None:
        base, ext = os.path.splitext(args.out_csv)
        args.summary_csv = f"{base}__summary.csv"
    summary.to_csv(args.summary_csv)

    # Pick representative example & plot (optional)
    if args.example_png:
        pick_col = "total_length_um"
        if args.pick == "median":
            target = df[pick_col].median()
            ex = df.iloc[(df[pick_col] - target).abs().argsort().iloc[0]]
        elif args.pick == "longest":
            ex = df.iloc[df[pick_col].idxmax()]
        else:
            ex = df.iloc[df[pick_col].idxmin()]
        os.makedirs(os.path.dirname(args.example_png), exist_ok=True)
        plot_swc_2d(ex["path"], args.example_png)
        print(f"Example SWC plotted: {ex['file']} → {args.example_png}")

    print(f"Saved per-neuron metrics: {args.out_csv}")
    print(f"Saved dataset summary:   {args.summary_csv}")

if __name__ == "__main__":
    main()
