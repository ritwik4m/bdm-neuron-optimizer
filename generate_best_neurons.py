import os
import subprocess
import shutil
import pandas as pd

# Best Trial 37 parameters
best_params = {
    "elong_apical": 31.61552084299452,
    "elong_basal": 37.132998804949594,
    "branch_apical": 0.027539048585432793,
    "branch_basal": 0.008885155100011196,
    "steps": 100,
    "length_scale": 1.1326860486819257,
    "taper_rate": 0.47501675711407176,
    "branch_angle": 101.31041899125114,
    "synapse_bias": 0.7408326280527571,
}

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "output")
param_file = os.path.join(script_dir, "bdm_params.txt")
raw_csv = os.path.join(output_dir, "raw_features.csv")

os.makedirs(output_dir, exist_ok=True)

# Write best parameters to file
with open(param_file, "w") as f:
    for k, v in best_params.items():
        f.write(f"{k} {v}\n")

# How many neurons to generate
n = 50
all_features = []

for i in range(n):
    print(f"\n🧠 Generating neuron {i+1}/{n}...")

    try:
        # Run simulator
        subprocess.run(["bdm", "run"], check=True, cwd=script_dir)

        # Rename output SWC
        swc_src = os.path.join(output_dir, "neuron_unknown.swc")
        swc_dst = os.path.join(output_dir, f"neuron_best_rep{i:02d}.swc")
        if not os.path.exists(swc_src):
            print("❌ neuron_unknown.swc not found!")
            continue
        shutil.move(swc_src, swc_dst)

        # Run feature extraction
        subprocess.run(["python", "extract_features.py", swc_dst], check=True, cwd=script_dir)

        # Save feature row
        if os.path.exists(raw_csv):
            df = pd.read_csv(raw_csv, nrows=1)
            if not df.empty:
                all_features.append(df.iloc[0])
            os.remove(raw_csv)
        else:
            print("⚠️ raw_features.csv not found after extraction.")

    except subprocess.CalledProcessError:
        print(f"❌ Simulation or extraction failed on rep {i}.")
        continue

# Combine and save all extracted features
if all_features:
    final_df = pd.DataFrame(all_features)
    final_df.to_csv(os.path.join(output_dir, "features_best_all.csv"), index=False)
    print("\n✅ All features saved to output/features_best_all.csv")
else:
    print("⚠️ No features extracted. Check logs.")
