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

import random
random.seed(42)
np.random.seed(42)

# === Set up dynamic paths ===
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

# === Load target features ===
if not os.path.exists(target_path):
    raise FileNotFoundError(f"❌ Target features not found: {target_path}")
with open(target_path) as f:
    target_features = json.load(f)

# === Load weights ===
def load_weights():
    default_weights = {key: 1.0 for key in target_features}
    if os.path.exists(weights_path):
        with open(weights_path) as wf:
            wcfg = json.load(wf)
        profile = wcfg.get("use_profile", "default")
        weights = wcfg.get("profiles", {}).get(profile, default_weights)
        print(f"[config] Using weights profile from {weights_path}: {profile}")
    else:
        print(f"[config] {weights_path} not found — using built-in defaults.")
        weights = default_weights
    return weights

weights = load_weights()

# === Distance function ===
def compute_loss(sim_row: dict, trial_id: int) -> float:
    loss = 0.0
    print(f"\n[trial {trial_id}] 🔎 Feature differences (normalized):")
    for key, t in target_features.items():
        if key not in sim_row or pd.isna(sim_row[key]):
            print(f"[trial {trial_id}] ❌ Missing or NaN for key: '{key}'")
            return float("inf")

        s = float(sim_row[key])
        w = float(weights.get(key, 1.0))

        rel_diff = (s - t) / t if abs(t) > 1e-6 else s - t
        rel_diff = np.clip(rel_diff, -2.0, 2.0)

        contrib = w * (rel_diff ** 2)
        loss += contrib

        print(
            f"  {key:25s} target={t:.3f}, sim={s:.3f}, "
            f"rel_diff={rel_diff:+.2%}, weighted_contrib={contrib:.4f}"
        )

    print(f"[trial {trial_id}] 🔥 Total normalized loss = {loss:.6f}")
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

    print(f"\n[trial {trial_id}] 🧪 Testing params:")
    print("  " + ", ".join(f"{k}={v:.3f}" for k, v in params.items()))

    # Write params to file
    with open(os.path.join(script_dir, "bdm_params.txt"), "w") as f:
        for k, v in params.items():
            f.write(f"{k} {v}\n")

    # Set up output file path
    swc_path = os.path.join(output_dir, f"neuron_{trial_id}.swc")
    os.environ["TRIAL_ID"] = str(trial_id)
    os.environ["SWC_PATH"] = swc_path

    # Run simulator
    try:
        subprocess.run(["bdm", "run"], check=True, cwd=script_dir)
    except subprocess.CalledProcessError:
        print(f"[trial {trial_id}] ❌ Simulation failed.")
        return float("inf")

    # Run feature extraction
    try:
        subprocess.run(["python", "extract_features.py"], check=True, cwd=script_dir)
    except subprocess.CalledProcessError:
        print(f"[trial {trial_id}] ❌ Feature extraction failed.")
        return float("inf")

    # Read features
    csv_path = os.path.join(output_dir, "raw_features.csv")
    try:
        df = pd.read_csv(csv_path, nrows=1, on_bad_lines="skip")
        if df.empty:
            print(f"[trial {trial_id}] ⚠️ raw_features.csv is empty.")
            return float("inf")

        # Save trial-specific version
        trial_features_path = os.path.join(output_dir, f"raw_features_{trial_id}.csv")
        df.to_csv(trial_features_path, index=False)
        sim_row = df.iloc[0].to_dict()

    except Exception as e:
        print(f"[trial {trial_id}] ❌ Failed to read raw_features.csv: {e}")
        return float("inf")

    # FIXED: wrap in list for compute_loss
    return compute_loss([sim_row], trial_id)


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

    def run_batch(n):
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
        b = run_batch(n)
        total_done = len(b)

        if total_done >= min_trials_before_stop:
            rel_imp = recent_relative_improvement(b, window=early_window)
            print(f"[monitor] Trials={total_done}, recent rel. improvement={rel_imp:.4%}")
            if rel_imp <= flat_tol:
                print(f"[monitor] Convergence has flattened (≤{flat_tol:.2%}). Stopping.")
                return total_done

    # Additional rounds if improving
    b = best_curve(study)
    rel_imp_20 = recent_relative_improvement(b, window=20)
    rel_imp_10 = recent_relative_improvement(b, window=10)
    extra = 100 if rel_imp_20 > 0.02 or rel_imp_10 > 0.015 else 50 if rel_imp_20 > 0.005 else 0

    if extra == 0:
        print("[monitor] Improvements are small; stopping.")
        return total_done

    print(f"[monitor] Still improving. Running +{extra} more trials.")
    target2 = total_done + extra

    while total_done < target2:
        n = min(batch_size, target2 - total_done)
        b = run_batch(n)
        total_done = len(b)
        rel_imp = recent_relative_improvement(b, window=early_window)
        print(f"[monitor] Trials={total_done}, recent rel. improvement={rel_imp:.4%}")
        if rel_imp <= flat_tol and total_done >= min_trials_before_stop:
            print(f"[monitor] Flattened during extension. Stopping early.")
            break

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
    try:
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "optuna_convergence.png"))
        plt.show()
        print("Saved convergence plot to output/optuna_convergence.png")
    except Exception as e:
        print(f"[plot] Failed to generate convergence plot: {e}")

    try:
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "param_importance.png"))
        plt.show()
        print("Saved parameter importance plot to output/param_importance.png")
    except Exception as e:
        print(f"[plot] Failed to generate importance plot: {e}")
