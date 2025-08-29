import os
import morphio
import pandas as pd
import numpy as np

# Use environment variable or fallback to default
SWC_FILE = os.getenv("SWC_PATH", "output/neuron.swc")
CSV_OUTPUT = "output/raw_features.csv"

def compute_max_depth(neuron):
    """Recursively compute max branch depth from soma using root sections."""
    def dfs(sec, current_depth):
        depths = [current_depth]
        for child in sec.children:
            depths.append(dfs(child, current_depth + 1))
        return max(depths)

    depths = []
    for sec in neuron.root_sections:
        if sec.type in (morphio.SectionType.basal_dendrite,
                        morphio.SectionType.apical_dendrite):
            depths.append(dfs(sec, 1))
    return max(depths) if depths else 0

def extract_features_morphio(swc_path):
    neuron = morphio.Morphology(swc_path)
    row = {}
    row['file_name'] = os.path.basename(swc_path)
    row['dendrites_only'] = 1

    segment_lengths = []
    bifurcations = 0
    tips = 0
    endpoints = []

    for sec in neuron.sections:
        if sec.type not in (morphio.SectionType.basal_dendrite,
                            morphio.SectionType.apical_dendrite):
            continue

        points = sec.points
        if len(points) > 1:
            seg_lens = np.linalg.norm(np.diff(points[:, :3], axis=0), axis=1)
            segment_lengths.extend(seg_lens)

        if len(sec.children) > 1:
            bifurcations += 1
        elif len(sec.children) == 0:
            tips += 1

        endpoints.extend(points[:, :3])

    total_length = float(np.sum(segment_lengths)) if segment_lengths else 0.0
    mean_len = float(np.mean(segment_lengths)) if segment_lengths else 0.0
    std_len = float(np.std(segment_lengths)) if segment_lengths else 0.0
    max_depth = compute_max_depth(neuron)

    row['total_length_um'] = total_length
    row['num_segments'] = len(segment_lengths)
    row['mean_segment_length_um'] = mean_len
    row['std_segment_length_um'] = std_len
    row['n_bifurcations'] = bifurcations
    row['n_tips'] = tips
    row['max_tree_depth'] = max_depth

    # Compute soma center and max radial extent before Sholl analysis
    soma_center = neuron.soma.center
    if endpoints:
        distances = np.linalg.norm(np.array(endpoints) - soma_center, axis=1)
        row['max_radial_extent_um'] = float(np.max(distances))
    else:
        row['max_radial_extent_um'] = 0.0

    # === Real Sholl analysis ===
    max_dist = row['max_radial_extent_um']
    shell_step = 5.0  # in µm
    radii = np.arange(0, max_dist + shell_step, shell_step)
    sholl_counts = np.zeros_like(radii)

    for sec in neuron.sections:
        if sec.type not in (morphio.SectionType.basal_dendrite,
                            morphio.SectionType.apical_dendrite):
            continue

        points = sec.points[:, :3]
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            d1 = np.linalg.norm(p1 - soma_center)
            d2 = np.linalg.norm(p2 - soma_center)

            for j, r in enumerate(radii):
                if (d1 - r) * (d2 - r) < 0:
                    sholl_counts[j] += 1

    row['sholl_peak'] = int(sholl_counts.max())
    row['sholl_peak_radius_um'] = float(radii[np.argmax(sholl_counts)]) if len(radii) > 0 else 0.0
    row['sholl_auc'] = int(np.sum(sholl_counts))

    len_100um = total_length / 100.0 if total_length else 1.0
    row['bif_per_100um'] = bifurcations / len_100um
    row['tips_per_100um'] = tips / len_100um

    return row

import sys

if __name__ == "__main__":
    # Support command-line argument (preferred)
    if len(sys.argv) > 1:
        SWC_FILE = sys.argv[1]
    else:
        # fallback to env or default
        SWC_FILE = os.getenv("SWC_PATH", "output/neuron.swc")

    print(f"📂 Reading SWC file: {SWC_FILE}")
    if not os.path.exists(SWC_FILE):
        print(f"❌ Error: SWC file not found at {SWC_FILE}")
        exit(1)

    features = extract_features_morphio(SWC_FILE)

    # 👇 Tag source: real vs synthetic
    features["source"] = "synthetic" if "neuron_" in features["file_name"] else "real"

    df = pd.DataFrame([features])
    os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"✅ Wrote extracted features to {CSV_OUTPUT}")
