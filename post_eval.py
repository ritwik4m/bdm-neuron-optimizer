import json, os, subprocess, pandas as pd, numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
TARGET_JSON = os.path.join(SCRIPT_DIR, "target_features.json")
RAW_CSV = os.path.join(OUTPUT_DIR, "raw_features.csv")
BDM_PARAMS = os.path.join(SCRIPT_DIR, "bdm_params.txt")

def load_best_params_from_log(trials_csv):
    df = pd.read_csv(trials_csv)
    df = df[df["state"]=="COMPLETE"].sort_values("value", ascending=True)
    best = df.iloc[0]
    # parameters are in columns like "params_branch_apical"
    params = {c.replace("params_", ""): best[c]
              for c in df.columns if c.startswith("params_")}
    return params

def write_params(params):
    with open(BDM_PARAMS, "w") as f:
        for k, v in params.items():
            f.write(f"{k} {v}\n")

def simulate_and_collect(n=20):
    rows = []
    for i in range(n):
        subprocess.run(["bdm", "run"], check=True, cwd=SCRIPT_DIR)
        subprocess.run(["python", "extract_features.py"], check=True, cwd=SCRIPT_DIR)
        df = pd.read_csv(RAW_CSV, nrows=1, on_bad_lines="skip")
        if not df.empty:
            rows.append(df.iloc[0].to_dict())
    return rows

def plot_overlaps(sim_rows, target_dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Collate sim distributions
    sim_by_key = {}
    for row in sim_rows:
        for k, v in row.items():
            sim_by_key.setdefault(k, []).append(v)

    # Only plot features present in target
    for k, tgt_vals in target_dict.items():
        if not isinstance(tgt_vals, list) or len(tgt_vals)==0:
            continue
        sim_vals = sim_by_key.get(k, [])
        if len(sim_vals) == 0:
            continue

        plt.figure()
        plt.hist(tgt_vals, bins=20, alpha=0.6, label="Target (SWC)")
        plt.hist(sim_vals, bins=20, alpha=0.6, label="Sim (best params)")
        plt.title(k)
        plt.xlabel(k)
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, f"overlap_{k}.png")
        plt.savefig(out, dpi=140)
        plt.close()

if __name__ == "__main__":
    # 1) load target distributions
    with open(TARGET_JSON) as f:
        target = json.load(f)

    # 2) read best params from trials log
    trials_csv = os.path.join(OUTPUT_DIR, "optuna_trials_log.csv")
    best_params = load_best_params_from_log(trials_csv)
    print("[best] parameters:", best_params)

    # 3) write them to bdm_params.txt
    write_params(best_params)

    # 4) generate N neurons with best params
    sim_rows = simulate_and_collect(n=30)
    print(f"[sim] collected {len(sim_rows)} rows")

    # 5) plot overlaps
    plot_overlaps(sim_rows, target)
    print("[ok] overlap plots saved in output/")
