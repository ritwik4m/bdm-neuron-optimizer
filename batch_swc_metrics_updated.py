# batch_swc_metrics_updated.py

import os
import json
import pandas as pd
from glob import glob
from tqdm import tqdm

# Import the correct function from swc_metrics.py
from swc_metrics import compute_features_from_swc


def build_target_distributions(
    swc_dir,
    out_json="target_features.json",
    out_csv="target_features.csv",
    recursive=False,
):
    """Process a directory of SWC files and extract morphology features into CSV + JSON."""
    feature_lists = {}

    # gather .swc files (case-insensitive), with optional recursion
    patterns = ["*.swc", "*.SWC"]
    if recursive:
        patterns = [os.path.join("**", p) for p in patterns]

    swc_files = []
    for p in patterns:
        swc_files.extend(glob(os.path.join(swc_dir, p), recursive=recursive))

    swc_files = sorted(set(swc_files))
    if not swc_files:
        raise FileNotFoundError(f"No SWC files found in {swc_dir}")

    print(f"[batch] Processing {len(swc_files)} SWC files from {swc_dir}...")

    rows = []
    for swc_path in tqdm(swc_files):
        try:
            feats = compute_features_from_swc(swc_path)

            # keep filename in the row to trace back if needed
            feats = {"swc_path": os.path.relpath(swc_path, swc_dir), **feats}
            rows.append(feats)

            # aggregate into lists per feature
            for k, v in feats.items():
                if k not in feature_lists:
                    feature_lists[k] = []
                feature_lists[k].append(v)

        except Exception as e:
            print(f"[batch] Failed to process {swc_path}: {e}")

    # Save as CSV (row per SWC file)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[batch] Saved feature table → {out_csv}")

    # Save as JSON (feature → list of values across all SWCs)
    with open(out_json, "w") as f:
        json.dump(feature_lists, f, indent=2)
    print(f"[batch] Saved distributional targets → {out_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True, help="Directory containing .swc files")
    parser.add_argument("--out_csv", required=True, help="Path to write the features CSV")
    parser.add_argument("--out_json", default="target_features.json", help="Path to write the JSON distributions")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories for .swc files")
    args = parser.parse_args()

    build_target_distributions(
        swc_dir=args.in_dir,
        out_json=args.out_json,
        out_csv=args.out_csv,
        recursive=args.recursive,
    )
