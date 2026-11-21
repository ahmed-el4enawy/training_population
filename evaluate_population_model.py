"""
evaluate_population_model.py

Evaluates the trained population-level neural state-space model
against the held-out simulated SIM002 TEST cohort (5 virtual subjects, 75 .pkl files,
825 seven-day trajectories, 27,225 non-overlapping 5-hour test episodes).

Computes clinical consensus metrics (TIR, TAR, TBR, LBGI, HBGI, MG) and
paired Two One-Sided Tests (TOST, alpha=0.05) against paper-reported equivalence margins.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from pickle import load
from typing import Dict, Tuple, Optional

import numpy as np
import scipy.stats as st
import torch

REPO_ROOT = Path(__file__).resolve().parent
T1DSIM_AI_SRC = REPO_ROOT / "T1DSim_AI" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(T1DSIM_AI_SRC) not in sys.path:
    sys.path.insert(0, str(T1DSIM_AI_SRC))

from t1dsim_ai.population_model import CGMOHSUSimStateSpaceModel_V2
from t1dsim_ai.options import n_neurons_pop
from t1dsim_ai.utils.preprocess import scale_inverse_Q1
from dataset_pipeline import (
    DEFAULT_DATASET_DIR,
    DEFAULT_CACHE_DIR,
    TRAJECTORY_LENGTH,
    TOTAL_PROBLEMS_PER_FILE,
    validate_cache,
    load_cached_split,
    parse_single_pkl_file,
)
from train_population_model import ForwardEulerSimulatorPop, FramedWindowSampler

# ==============================================================================
# CONFIG & REFERENCE BENCHMARKS
# ==============================================================================
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "PopulationModel" / "population_model_05022024_epoch_15.pt"
DEFAULT_SCALER_DIR = REPO_ROOT / "models" / "PopulationModel"

SEQ_LEN = 61
TEST_OVERLAP = 0.0  # Zero overlap = non-overlapping 5-hour episodes (hop=60)
EVAL_BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TIR_LOW, TIR_HIGH = 70.0, 180.0
MARGINS = {"TIR": 5.0, "TAR": 5.0, "TBR": 1.0, "LBGI": 0.5, "HBGI": 1.0, "MG": 10.0}
ALPHA = 0.05

PAPER_REFERENCE_METRICS = {
    "TIR": {"mean": 72.9, "std": 35.2, "unit": "%"},
    "TAR": {"mean": 23.1, "std": 36.3, "unit": "%"},
    "TBR": {"mean": 4.0, "std": 8.2, "unit": "%"},
    "LBGI": {"mean": 1.5, "std": 1.7, "unit": ""},
    "HBGI": {"mean": 4.2, "std": 6.2, "unit": ""},
    "MG": {"mean": 132.6, "std": 48.2, "unit": "mg/dL"},
}


# ==============================================================================
# Clinical Outcome Metrics (Consensus & Risk Indices)
# ==============================================================================
def window_outcomes_batch(bg: np.ndarray) -> Dict[str, np.ndarray]:
    tir = 100.0 * np.mean((bg >= TIR_LOW) & (bg <= TIR_HIGH), axis=1)
    tar = 100.0 * np.mean(bg > TIR_HIGH, axis=1)
    tbr = 100.0 * np.mean(bg < TIR_LOW, axis=1)

    bgc = np.clip(bg, 1.0, None)
    f_bg = 1.509 * (np.log(bgc) ** 1.084 - 5.381)
    r_bg = 10.0 * (f_bg ** 2)
    neg = f_bg < 0
    pos = ~neg
    n = bg.shape[1]
    lbgi = (r_bg * neg).sum(axis=1) / n
    hbgi = (r_bg * pos).sum(axis=1) / n

    return {"TIR": tir, "TAR": tar, "TBR": tbr, "LBGI": lbgi, "HBGI": hbgi, "MG": bg.mean(axis=1)}


def paired_tost(sim: np.ndarray, actual: np.ndarray, margin: float, alpha: float = ALPHA) -> Tuple:
    diff = sim - actual
    n = len(diff)
    if n < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"), 1.0, False)

    mean_diff = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n))

    if se == 0:
        p_value = 0.0 if abs(mean_diff) < margin else 1.0
        return (
            float(sim.mean()), float(sim.std(ddof=1)),
            float(actual.mean()), float(actual.std(ddof=1)),
            p_value, p_value < alpha
        )

    t1 = (mean_diff + margin) / se
    p1 = 1.0 - float(st.t.cdf(t1, df=n - 1))
    t2 = (mean_diff - margin) / se
    p2 = float(st.t.cdf(t2, df=n - 1))

    p_value = max(p1, p2)
    return (
        float(sim.mean()), float(sim.std(ddof=1)),
        float(actual.mean()), float(actual.std(ddof=1)),
        p_value, p_value < alpha
    )


# ==============================================================================
# Evaluation Execution
# ==============================================================================
def evaluate(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    scaler_dir: Path = DEFAULT_SCALER_DIR,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    batch_size: int = EVAL_BATCH_SIZE,
    output_json: Optional[Path] = None,
    output_report: Optional[Path] = None,
) -> Dict:
    checkpoint_path = Path(checkpoint_path).resolve()
    scaler_dir = Path(scaler_dir).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    cache_dir = Path(cache_dir).resolve()

    print("=" * 70)
    print("SIM002 HELD-OUT TEST EVALUATION")
    print(f"Model Checkpoint: {checkpoint_path}")
    print(f"Scaler Directory: {scaler_dir}")
    print(f"Device:           {DEVICE}")
    print("=" * 70)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    # Load scalers
    scaler_states_path = scaler_dir / "scaler_states.pkl"
    scaler_inputs_path = scaler_dir / "scaler_inputs.pkl"
    if not scaler_states_path.exists() or not scaler_inputs_path.exists():
        raise FileNotFoundError(f"Scaler files missing in {scaler_dir}")

    with open(scaler_states_path, "rb") as fh:
        scaler_states = load(fh)
    with open(scaler_inputs_path, "rb") as fh:
        scaler_inputs = load(fh)

    # Load test split
    valid_cache, manifest = validate_cache(cache_dir)
    if valid_cache and "shapes" in manifest and "TEST_states" in manifest["shapes"]:
        print(f"Loading TEST data from memory-mapped cache: {cache_dir}")
        raw_x_test, raw_u_test, _ = load_cached_split(cache_dir, "TEST", mode="r")
        n_rows = raw_x_test.shape[0]
        n_traj = n_rows // TRAJECTORY_LENGTH
        x_raw_all = np.array(raw_x_test, dtype=np.float32)
        u_raw_all = np.array(raw_u_test, dtype=np.float32)
    else:
        test_dir = dataset_dir / "SIM002" / "POPULATION" / "TEST"
        print(f"Loading TEST data directly from .pkl files: {test_dir}")
        test_files = sorted(list(test_dir.glob("*.pkl")))
        if len(test_files) == 0:
            raise FileNotFoundError(f"No .pkl test files found in {test_dir}")

        x_list, u_list = [], []
        for f in test_files:
            x_f, u_f, _ = parse_single_pkl_file(f, is_sim005=False)
            x_list.append(x_f)
            u_list.append(u_f)

        x_stacked = np.concatenate(x_list, axis=0)  # (N_traj, 2016, 10)
        u_stacked = np.concatenate(u_list, axis=0)  # (N_traj, 2016, 2)
        n_traj = x_stacked.shape[0]
        x_raw_all = x_stacked.reshape(-1, 10)
        u_raw_all = u_stacked.reshape(-1, 2)

    print(f"Loaded {n_traj:,} TEST trajectories ({x_raw_all.shape[0]:,} timesteps).")

    # Scale test data
    x_scaled = scaler_states.transform(x_raw_all).astype(np.float32)
    u_scaled = scaler_inputs.transform(u_raw_all).astype(np.float32)

    x_test_3d = x_scaled.reshape(n_traj, TRAJECTORY_LENGTH, 10)
    u_test_3d = u_scaled.reshape(n_traj, TRAJECTORY_LENGTH, 2)

    # Load model
    print(f"Loading model state dict from {checkpoint_path.name}...")
    model = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    simulator = ForwardEulerSimulatorPop(model, ts=5.0).to(DEVICE)
    sampler = FramedWindowSampler(
        x_test_3d, u_test_3d, SEQ_LEN, overlap=TEST_OVERLAP, batch_size=batch_size,
        device=DEVICE, seed=0
    )
    all_pairs = sampler.pairs
    n_windows = len(all_pairs)
    print(f"Evaluating {n_windows:,} non-overlapping 5-hour test episodes (hop={sampler.hop})...")

    sim_glucose_list = []
    actual_glucose_list = []

    t0 = time.time()
    with torch.no_grad():
        for b_start in range(0, n_windows, batch_size):
            b_end = min(b_start + batch_size, n_windows)
            batch_pairs = all_pairs[b_start:b_end]

            x0, u_batch, x_true = sampler.batch_from_pairs(batch_pairs)
            x_sim = simulator(x0, u_batch)

            # Extract Q1 (channel 0) and invert scaling to mg/dL
            y_sim_scaled = x_sim.cpu().numpy()[:, :, 0].T   # (B, SEQ_LEN)
            y_true_scaled = x_true.cpu().numpy()[:, :, 0].T # (B, SEQ_LEN)

            y_sim_mgdl = y_sim_scaled * scaler_states.scale_[0] + scaler_states.center_[0]
            y_true_mgdl = y_true_scaled * scaler_states.scale_[0] + scaler_states.center_[0]

            sim_glucose_list.append(y_sim_mgdl)
            actual_glucose_list.append(y_true_mgdl)

    print(f"Simulation completed in {time.time() - t0:.2f}s.")

    sim_bg = np.concatenate(sim_glucose_list, axis=0)
    actual_bg = np.concatenate(actual_glucose_list, axis=0)

    sim_outcomes = window_outcomes_batch(sim_bg)
    actual_outcomes = window_outcomes_batch(actual_bg)

    print("\n" + "=" * 80)
    print("TOST STATISTICAL EQUIVALENCE RESULTS (alpha=0.05)")
    print("=" * 80)
    print(f"{'Metric':<10} {'Margin':<8} {'Paper Ref':<20} {'Local ODE':<20} {'NN Digital Twin':<20} {'TOST p':<10} {'Equiv?'}")
    print("-" * 80)

    results = {}
    for metric in ["TIR", "TAR", "TBR", "LBGI", "HBGI", "MG"]:
        sim_val = sim_outcomes[metric]
        act_val = actual_outcomes[metric]
        margin = MARGINS[metric]

        s_m, s_s, a_m, a_s, p_val, is_eq = paired_tost(sim_val, act_val, margin)
        ref = PAPER_REFERENCE_METRICS[metric]
        ref_str = f"{ref['mean']:.1f} +/- {ref['std']:.1f} {ref['unit']}".strip()
        local_str = f"{a_m:.1f} +/- {a_s:.1f} {ref['unit']}".strip()
        sim_str = f"{s_m:.1f} +/- {s_s:.1f} {ref['unit']}".strip()

        print(f"{metric:<10} +/-{margin:<6} {ref_str:<20} {local_str:<20} {sim_str:<20} {p_val:<10.2e} {'YES' if is_eq else 'NO'}")

        results[metric] = {
            "margin": margin,
            "paper_reference": ref,
            "local_ode_mean": a_m,
            "local_ode_std": a_s,
            "nn_model_mean": s_m,
            "nn_model_std": s_s,
            "tost_p_value": p_val,
            "equivalent": bool(is_eq),
        }

    # Save JSON and Report
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved machine-readable results to {output_json}")

    if output_report:
        generate_markdown_report(results, n_windows, n_traj, output_report)
        print(f"Saved evaluation markdown report to {output_report}")

    return results


def generate_markdown_report(results: Dict, n_windows: int, n_traj: int, out_path: Path):
    md = f"""# SIM002 TEST Evaluation Report

**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  
**Evaluated Cohort:** SIM002 Held-Out TEST (5 virtual subjects, {n_traj:,} seven-day trajectories)  
**Evaluated Windows:** {n_windows:,} non-overlapping 5-hour episodes (SEQ_LEN=61, hop=60)  

---

## TOST Statistical Equivalence (alpha=0.05)

| Metric | Equivalence Margin | Published Paper ODE | Local ODE Ground Truth | NN Digital Twin | TOST p-value | Equivalent? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for m in ["TIR", "TAR", "TBR", "LBGI", "HBGI", "MG"]:
        r = results[m]
        ref = r["paper_reference"]
        ref_s = f"{ref['mean']:.1f} +/- {ref['std']:.1f} {ref['unit']}".strip()
        loc_s = f"{r['local_ode_mean']:.1f} +/- {r['local_ode_std']:.1f} {ref['unit']}".strip()
        nn_s = f"{r['nn_model_mean']:.1f} +/- {r['nn_model_std']:.1f} {ref['unit']}".strip()
        eq_s = "**YES**" if r["equivalent"] else "**NO**"
        md += f"| **{m}** | $\\pm {r['margin']}$ | {ref_s} | {loc_s} | {nn_s} | `{r['tost_p_value']:.2e}` | {eq_s} |\n"

    out_path.write_text(md, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Population Model on SIM002 TEST")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Model checkpoint path")
    parser.add_argument("--scaler-dir", type=str, default=str(DEFAULT_SCALER_DIR), help="Path containing scalers")
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR), help="Dataset root directory")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR), help="Cache directory")
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE, help="Batch size")
    parser.add_argument("--output-json", type=str, default=None, help="Output JSON path")
    parser.add_argument("--output-report", type=str, default=None, help="Output Markdown report path")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=Path(args.checkpoint),
        scaler_dir=Path(args.scaler_dir),
        dataset_dir=Path(args.dataset_dir),
        cache_dir=Path(args.cache_dir),
        batch_size=args.batch_size,
        output_json=Path(args.output_json) if args.output_json else None,
        output_report=Path(args.output_report) if args.output_report else None,
    )


if __name__ == "__main__":
    main()
