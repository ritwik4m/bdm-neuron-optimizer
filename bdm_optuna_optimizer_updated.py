import optuna
import subprocess
import pandas as pd
import numpy as np
import json
import os
import shutil
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from optuna.visualization import matplotlib
from scipy.stats import wasserstein_distance, iqr

import random
random.seed(42)
np.random.seed(42)

# === Setup paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
target_path  = os.path.join(script_dir, "target_features.json")
weights_path = os.path.join(script_dir, "weights.json")
output_dir   = os.path.join(script_dir, "output")

# === Clean old outputs ===
if os.path.exists(output_dir):
    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
        else:
            shutil.rmtree(fpath)
    print("[init] Cleaned old files in output/")
else:
    os.makedirs(output_dir)

# === Load target distributions ===
if not os.path.exists(target_path):
    raise FileNotFoundError(f"❌ Target features not found: {target_path}")
with open(target_path) as f:
    target_distributions = json.load(f)

# === Load weights ===
def load_weights():
    default_weights = {key: 1.0 for key in target_distributions}
    if os.path.exists(weights_path):
        with open(weights_path) as wf:
            wcfg = json.load(wf)
        profile = wcfg.get("use_profile", "default")
        weights = wcfg.get("profiles", {}).get(profile, default_weights)
        print(f"[config] Using weights profile from {weights_path}: {profile}")
    else:
        print(f"[config] {weights_path} not found — using defaults (all 1.0).")
        weights = default_weights
    return weights

weights = load_weights()

# === Helper: robust scaling for distributions ===
def _robust_scale(vals):
    vals = np.asarray(vals, dtype=float)
    med = np.median(vals)
    scale = iqr(vals) if iqr(vals) > 1e-9 else (np.std(vals) or 1.0)
    return (vals - med) / (scale if scale != 0 else 1.0)

# === Distance function (distributional loss) ===
def compute_loss(sim_rows, trial_id: int) -> float:
    loss = 0.0
    print(f"\n[trial {trial_id}] 🔎 Distributional differences (Wasserstein, robust-scaled):")
    for key, tgt_vals in target_distributions.items():
        if not tgt_vals:
            continue

        sim_vals = [row.get(key) for row in sim_rows if key in row and not pd.isna(row[key])]
        if len(sim_vals) == 0:
            print(f"[trial {trial_id}] ❌ Missing values for feature '{key}'")
            return float("inf")

        # scale both distributions
        sim_s = _robust_scale(sim_vals)
        tgt_s = _robust_scale(tgt_vals)

        d = wasserstein_distance(sim_s, tgt_s)
        w = float(weights.get(key, 1.0))
        contrib = w * d
        loss += contrib

        print(f"  {key:25s} W={d:.4f}, w={w:.2f}, contrib={contrib:.4f}")

    print(f"[trial {trial_id}] 🔥 Total loss = {loss:.6f}")
    return loss

# === Objective ===
def objective(trial):
    trial_id = trial.number
    params = {
        "elong_apical":   trial.suggest_float("elong_apical", 25.0, 45.0),
        "elong_basal":    trial.suggest_float("elong_basal", 25.0, 40.0),
        "branch_apical":  trial.suggest_float("branch_apical", 0.025, 0.04),
        "branch_basal":   trial.suggest_float("branch_basal", 0.007, 0.015),
        "steps":          trial.suggest_int("steps", 90, 120),
        "length_scale":   trial.suggest_float("length_scale", 0.5, 2.0),
        "taper_rate":     trial.suggest_float("taper_rate", 0.1, 1.2),
        "branch_angle":   trial.suggest_float("branch_angle", 20.0, 110.0),
        "synapse_bias":   trial.suggest_float("synapse_bias", 0.0, 1.0),
    }

    print(f"\n[trial {trial_id}] 🧪 Testing params: "
          + ", ".join(f"{k}={v:.3f}" for k, v in params.items()))

    # Write params to file for the simulator
    with open(os.path.join(script_dir, "bdm_params.txt"), "w") as f:
        for k, v in params.items():
            f.write(f"{k} {v}\n")

    # Save params per trial (optional, for logging/debug)
    param_path = os.path.join(output_dir, f"params_trial{trial_id:04d}.json")
    with open(param_path, "w") as pf:
        json.dump(params, pf, indent=2)

    sim_rows = []
    n_reps = 5  # number of neurons per trial
    for rep in range(n_reps):
        try:
            # Run simulation
            subprocess.run(["bdm", "run"], check=True, cwd=script_dir)

            # Rename the output SWC to something unique
            swc_src = os.path.join(output_dir, "neuron_unknown.swc")
            swc_dst = os.path.join(output_dir, f"neuron_trial{trial_id:04d}_rep{rep:02d}.swc")

            if not os.path.exists(swc_src):
                print(f"[trial {trial_id}] ❌ SWC file not found after simulation: {swc_src}")
                return float("inf")

            shutil.move(swc_src, swc_dst)

            # Run feature extraction, passing the renamed SWC file
            subprocess.run(["python", "extract_features.py", swc_dst], check=True, cwd=script_dir)

            # Read extracted features (assuming always saved to raw_features.csv)
            df = pd.read_csv(os.path.join(output_dir, "raw_features.csv"),
                             nrows=1, on_bad_lines="skip")

            if not df.empty:
                sim_rows.append(df.iloc[0].to_dict())

        except subprocess.CalledProcessError:
            print(f"[trial {trial_id}] ❌ Rep {rep} failed.")
            return float("inf")
        except Exception as e:
            print(f"[trial {trial_id}] ⚠️ Unexpected error: {e}")
            return float("inf")

    if not sim_rows:
        return float("inf")

    return compute_loss(sim_rows, trial_id)


# === Convergence utilities ===
def best_curve(study):
    vals, cur = [], math.inf
    for t in study.trials:
        v = t.value if t.value is not None and not math.isnan(t.value) else cur
        cur = min(cur, v)
        vals.append(cur)
    return vals

def recent_relative_improvement(best_vals, window=20):
    if len(best_vals) < window + 1:
        return 0.0
    prev, cur = best_vals[-window-1], best_vals[-1]
    if prev in (None, math.inf) or cur in (None, math.inf):
        return 0.0
    return max(0.0, (prev - cur) / max(abs(prev), 1.0))

def save_convergence_plot(best_vals, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(best_vals)+1), best_vals, marker=".", linewidth=1)
    plt.xlabel("Trial")
    plt.ylabel("Running best (loss)")
    plt.title("Optuna Convergence")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def optimize_in_batches(study, objective, start_trials=100, batch_size=20,
                        early_window=20, flat_tol=0.005, min_trials_before_stop=40):
    total_target = start_trials
    total_done = len(study.trials)

    def run_batch(n, done):
        print(f"\n🚀 Running batch of {n} trials (completed: {done}/{total_target})")
        study.optimize(objective, n_trials=n, catch=(Exception,))
        try:
            study.trials_dataframe().to_csv(os.path.join(output_dir, "optuna_trials_log.csv"), index=False)
        except Exception:
            pass
        b = best_curve(study)
        save_convergence_plot(b, path=os.path.join(output_dir, "convergence.png"))
        return b

    while total_done < total_target:
        n = min(batch_size, total_target - total_done)
        b = run_batch(n, total_done)
        total_done = len(b)

        print(f"✅ Total completed trials: {total_done}/{total_target}")

        if total_done >= min_trials_before_stop:
            rel_imp = recent_relative_improvement(b, window=early_window)
            print(f"[monitor] Trials={total_done}, recent rel. improvement={rel_imp:.4%}")
            if rel_imp <= flat_tol:
                print(f"[monitor] Convergence has flattened (≤{flat_tol:.2%}). Stopping early.")
                return total_done

    return total_done


# === Main ===
if __name__ == "__main__":
    storage_path = "sqlite:///output/bdm_study.db"
    study = optuna.create_study(
        direction="minimize",
        study_name="bdm_opt",
        storage=storage_path,
        load_if_exists=True
    )

    total_trials = optimize_in_batches(
        study,
        objective,
        start_trials=100,
        batch_size=20,
        early_window=20,
        flat_tol=0.05,
        min_trials_before_stop=60
    )

    print(f"\n[summary] Completed {total_trials} trials.")
    print("\nBest trial:")
    print(study.best_trial)
    print("\nBest parameters:")
    print(study.best_trial.params)

    # Save Optuna plots
   # Save Optuna plots
try:
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optuna_convergence.png"))
    plt.close()
except Exception as e:
    print(f"[plot] Failed to generate convergence plot: {e}")

try:
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "param_importance.png"))
    plt.close()
except Exception as e:
    print(f"[plot] Failed to generate importance plot: {e}")

