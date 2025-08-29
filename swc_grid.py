# swc_grid.py
import os, re, glob
import matplotlib.pyplot as plt

SWC_DIR = "/workspace/pyramidal_cell/output"

def read_swc_edges(path):
    ids = {}
    parents = {}
    xs, ys = {}, {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): 
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            nid = int(float(parts[0]))
            x = float(parts[2]); y = float(parts[3])
            parent = int(float(parts[6]))
            ids[nid] = True
            xs[nid] = x; ys[nid] = y
            parents[nid] = parent

    root_ids = [n for n,p in parents.items() if p == -1 or p == n or p == 0]
    if root_ids:
        cx = xs[root_ids[0]]; cy = ys[root_ids[0]]
    else:
        cx = sum(xs.values())/len(xs) if xs else 0.0
        cy = sum(ys.values())/len(ys) if ys else 0.0

    segments = []
    for n, p in parents.items():
        if p in (-1, 0) or p not in xs or n not in xs:
            continue
        segments.append((xs[n]-cx, ys[n]-cy, xs[p]-cx, ys[p]-cy))
    return segments

def numeric_sort_key(fname):
    m = re.search(r"rep(\d+)", fname)
    return int(m.group(1)) if m else 0

def plot_swc_grid(pattern, out_png, cols=10, lw=0.5):
    files = sorted(glob.glob(os.path.join(SWC_DIR, pattern)), key=numeric_sort_key)
    if not files:
        print(f"No files match {pattern} in {SWC_DIR}")
        return
    n = len(files)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.6, rows*1.6))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, path in enumerate(files):
        ax = axes[idx]
        segs = read_swc_edges(path)
        for x1, y1, x2, y2 in segs:
            ax.plot([x1, x2], [y1, y2], linewidth=lw, color="black")
        ax.set_title(os.path.basename(path).replace(".swc",""), fontsize=6)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    for j in range(idx+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(SWC_DIR, out_png), dpi=300)
    plt.close()
    print(f"✅ Saved {out_png} with {n} neurons.")

if __name__ == "__main__":
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"neuron_manual_grid_{timestamp}.png"
    plot_swc_grid("neuron_manual_rep*.swc", out_name)
