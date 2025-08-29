import os
import matplotlib.pyplot as plt

def load_swc(filepath):
    nodes = []
    with open(filepath) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            x, y = float(parts[2]), float(parts[3])
            nodes.append((x, y))
    return nodes

# Folder with SWC files
swc_folder = "/workspace/pyramidal_cell/output"

# Get all neuron_best_rep*.swc files
swc_files = sorted([
    f for f in os.listdir(swc_folder)
    if f.startswith("neuron_best_rep") and f.endswith(".swc")
])
n_files = len(swc_files)
cols = 10
rows = (n_files + cols - 1) // cols

# Plot grid
fig, axes = plt.subplots(rows, cols, figsize=(cols*1.8, rows*1.8))
axes = axes.flatten()

for idx, fname in enumerate(swc_files):
    path = os.path.join(swc_folder, fname)
    coords = load_swc(path)
    if coords:
        xs, ys = zip(*coords)
        axes[idx].plot(xs, ys, 'k-', linewidth=0.5)
    axes[idx].set_title(f"rep{idx:02d}", fontsize=6)
    axes[idx].axis('off')

# Hide unused subplots
for j in range(idx + 1, len(axes)):
    axes[j].axis('off')

# Save the figure instead of showing it
output_path = os.path.join(swc_folder, "neuron_grid.png")
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()
