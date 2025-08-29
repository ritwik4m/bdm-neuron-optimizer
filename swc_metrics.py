# swc_metrics.py

import pandas as pd
import numpy as np
import os

def compute_features_from_swc(swc_file, shell_step=5.0):
    """
    Compute morphological features from a SWC file.
    Returns a dictionary with feature names and values.
    """

    def _nan_result():
        keys = [
            "total_length_um", "num_segments", "mean_segment_length_um", "std_segment_length_um",
            "n_bifurcations", "n_tips", "max_tree_depth", "sholl_peak",
            "sholl_peak_radius_um", "sholl_auc", "max_radial_extent_um",
            "bif_per_100um", "tips_per_100um"
        ]
        return {k: float('nan') for k in keys}

    try:
        df = pd.read_csv(
            swc_file,
            comment="#",
            sep=r"\s+",
            header=None,
            names=["id", "type", "x", "y", "z", "radius", "parent"],
            engine="python",
        )

        if df.empty:
            raise ValueError(f"SWC file is empty: {swc_file}")

        df["id"] = df["id"].astype(int)
        df["parent"] = df["parent"].astype(int)
        df = df.set_index("id", drop=False)

        # --- Parent join (safe) ---
        seg_mask = df["parent"] != -1
        df_seg = df.loc[seg_mask].join(
            df[["x", "y", "z"]].rename(columns={"x": "x_p", "y": "y_p", "z": "z_p"}),
            how="left",
            on="parent"
        )

        df_seg["dist"] = np.linalg.norm(
            df_seg[["x", "y", "z"]].to_numpy() -
            df_seg[["x_p", "y_p", "z_p"]].to_numpy(),
            axis=1
        )
        df_seg["dist"] = df_seg["dist"].fillna(0.0)

        df["dist"] = 0.0
        df.loc[df_seg.index, "dist"] = df_seg["dist"].to_numpy()

        # --- Segment-length metrics ---
        seg_dists = df.loc[seg_mask, "dist"].astype(float)
        total_length_um = float(np.nansum(seg_dists))
        num_segments = int(seg_mask.sum())
        mean_segment_length_um = float(np.nanmean(seg_dists)) if num_segments else float("nan")
        std_segment_length_um = float(np.nanstd(seg_dists, ddof=1)) if num_segments > 1 else float("nan")

        # --- Tips & bifurcations ---
        child_counts = df.loc[seg_mask, "parent"].value_counts()
        child_counts = child_counts.reindex(df["id"], fill_value=0)
        is_root = df["parent"] == -1
        tip_mask = (child_counts == 0) & (~is_root)
        n_tips = int(tip_mask.sum())
        n_bifurcations = int((child_counts >= 2).sum())

        # --- Soma ---
        if (df["type"] == 1).any():
            soma_pts = df.loc[df["type"] == 1, ["x", "y", "z"]].to_numpy()
        elif is_root.any():
            soma_pts = df.loc[is_root, ["x", "y", "z"]].to_numpy()
        else:
            soma_pts = df.iloc[[0]][["x", "y", "z"]].to_numpy()
        soma = soma_pts.mean(axis=0)

        # --- Radial distances ---
        coords = df[["x", "y", "z"]].to_numpy()
        radial = np.linalg.norm(coords - soma, axis=1)
        df["radial_distance"] = radial
        max_radial_extent_um = float(np.nanmax(radial))

        # --- Tree depth (robust) ---
        depth = pd.Series(index=df.index, dtype=float)
        depth[is_root.values] = 0.0
        safety = 0
        updated = True
        while updated and safety < len(df) + 5000:
            updated = False
            safety += 1
            for node_id in df.index[depth.isna()]:
                parent_id = df.at[node_id, "parent"]
                if parent_id in depth.index and pd.notna(depth.at[parent_id]):
                    depth.at[node_id] = depth.at[parent_id] + 1.0
                    updated = True
        max_tree_depth = int(depth.max()) if depth.notna().any() else 0

        # --- Sholl analysis (row-wise safe) ---
        seg_tbl = df.loc[seg_mask, ["id", "parent", "radial_distance"]].copy()
        seg_tbl = seg_tbl.join(
            df["radial_distance"].rename("radial_distance_p"),
            on="parent", how="left"
        )
        seg_tbl = seg_tbl.dropna(subset=["radial_distance", "radial_distance_p"])

        sholl_peak = 0
        sholl_peak_radius_um = 0.0
        sholl_auc = 0.0
        if max_radial_extent_um > 0 and not seg_tbl.empty:
            radii = np.arange(shell_step, max_radial_extent_um + shell_step, shell_step)
            sholl_counts = []
            for r in radii:
                crossings = 0
                for _, row in seg_tbl.iterrows():
                    rc, rp = row["radial_distance"], row["radial_distance_p"]
                    if (rc - r) * (rp - r) < 0:  # straddles radius
                        crossings += 1
                    elif (abs(rc - r) < 1e-9) ^ (abs(rp - r) < 1e-9):  # exact hit
                        crossings += 1
                sholl_counts.append(crossings)

            sholl_counts = np.asarray(sholl_counts, dtype=int)
            sholl_peak = int(sholl_counts.max()) if sholl_counts.size else 0
            sholl_peak_radius_um = float(radii[int(sholl_counts.argmax())]) if sholl_counts.size else 0.0
            sholl_auc = float(np.sum(sholl_counts) * shell_step)

        # --- Rates per 100 µm ---
        bif_per_100um = float((n_bifurcations / total_length_um) * 100) if total_length_um > 0 else 0.0
        tips_per_100um = float((n_tips / total_length_um) * 100) if total_length_um > 0 else 0.0

        return {
            "total_length_um": total_length_um,
            "num_segments": num_segments,
            "mean_segment_length_um": mean_segment_length_um,
            "std_segment_length_um": std_segment_length_um,
            "n_bifurcations": n_bifurcations,
            "n_tips": n_tips,
            "max_tree_depth": max_tree_depth,
            "sholl_peak": sholl_peak,
            "sholl_peak_radius_um": sholl_peak_radius_um,
            "sholl_auc": sholl_auc,
            "max_radial_extent_um": max_radial_extent_um,
            "bif_per_100um": bif_per_100um,
            "tips_per_100um": tips_per_100um,
        }

    except Exception as e:
        print(f"[!] Failed to process SWC file {swc_file}: {e}")
        return _nan_result()


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python swc_metrics.py path/to/neuron.swc")
        sys.exit(1)
    swc_path = sys.argv[1]
    features = compute_features_from_swc(swc_path)
    print(json.dumps(features, indent=2))
