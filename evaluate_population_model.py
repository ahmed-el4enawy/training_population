"""
eval_population_model.py  (fixed version)

Evaluates the trained population NN model on the held-out TEST split and
reproduces the population-level rows of Table 3 (arXiv:2508.05705):
per-5-hour-window TIR/TAR/TBR/LBGI/HBGI/MG for simulated vs. actual, with
paired TOST equivalence tests.

Run after train_population_model.py (same folder, same OUTPUT_DIR).
"""

import os
from pickle import load

import numpy as np
import torch
import scipy.stats as st

from t1dsim_ai.population_model import CGMOHSUSimStateSpaceModel_V2
from t1dsim_ai.options import n_neurons_pop
from t1dsim_ai.utils.preprocess import scale_inverse_Q1, scale_single_state

from train_population_model import (
    _detect_backend,
    get_dataset_dims,
    group_split,
    fill_split_buffer,
    transform_in_place,
    FramedWindowSampler,
    ForwardEulerSimulatorPop,
    MERGED_MAT_PATH,
    OUTPUT_DIR,
    MODEL_FILENAME,
    SEED,
    TRAIN_FRACTION,
    VAL_FRACTION,
    SEQ_LEN,
    INPUT_COLUMN_PERM,
)

# ==============================================================================
# CONFIG
# ==============================================================================
EVAL_BATCH_SIZE = 256
MAX_TEST_SCENARIOS = 10000   # traces loaded for evaluation (whole groups of 5). None = all (~4.5 GB)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TIR_LOW, TIR_HIGH = 70.0, 180.0
MARGINS = {"TIR": 5.0, "TAR": 5.0, "TBR": 1.0, "LBGI": 0.5, "HBGI": 1.0, "MG": 10.0}
ALPHA = 0.05


# ==============================================================================
# Clinical outcome metrics (vectorized over windows)
# ==============================================================================
def window_outcomes_batch(bg):
    """bg: (q, m) array, one 5-hour window per row, mg/dL.
    Returns dict of (q,) arrays for TIR/TAR/TBR/LBGI/HBGI/MG."""
    tir = 100.0 * np.mean((bg >= TIR_LOW) & (bg <= TIR_HIGH), axis=1)
    tar = 100.0 * np.mean(bg > TIR_HIGH, axis=1)
    tbr = 100.0 * np.mean(bg < TIR_LOW, axis=1)

    bgc = np.clip(bg, 1.0, None)
    f_bg = 1.509 * (np.log(bgc) ** 1.084 - 5.381)
    r_bg = 10.0 * f_bg ** 2
    neg = f_bg < 0
    pos = ~neg
    lbgi = np.where(neg.any(axis=1), (r_bg * neg).sum(axis=1) / np.maximum(neg.sum(axis=1), 1), 0.0)
    hbgi = np.where(pos.any(axis=1), (r_bg * pos).sum(axis=1) / np.maximum(pos.sum(axis=1), 1), 0.0)

    return {"TIR": tir, "TAR": tar, "TBR": tbr, "LBGI": lbgi, "HBGI": hbgi, "MG": bg.mean(axis=1)}


# ==============================================================================
# Paired TOST equivalence test
# ==============================================================================
def paired_tost(sim, actual, margin, alpha=ALPHA):
    """Two one-sided paired t-tests for equivalence within +/-margin.
    p_value = larger of the two one-sided p-values (paper convention)."""
    diff = sim - actual
    n = len(diff)
    mean_diff = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    if se == 0:
        se = 1e-12

    t1 = (mean_diff + margin) / se
    p1 = 1 - st.t.cdf(t1, df=n - 1)
    t2 = (mean_diff - margin) / se
    p2 = st.t.cdf(t2, df=n - 1)

    p_value = max(p1, p2)
    return (sim.mean(), sim.std(ddof=1), actual.mean(), actual.std(ddof=1),
            p_value, p_value < alpha)


# ==============================================================================
# Main
# ==============================================================================
def main():
    print(f"Loading dataset dims from: {MERGED_MAT_PATH}")
    backend = _detect_backend(MERGED_MAT_PATH)
    n_scenarios, T, n_states, n_inputs = get_dataset_dims(MERGED_MAT_PATH, backend)

    print("Re-deriving train/val/TEST split (identical to training) ...")
    _, _, test_idx = group_split(
        n_scenarios, group_size=5, train_frac=TRAIN_FRACTION, val_frac=VAL_FRACTION, seed=SEED
    )
    if MAX_TEST_SCENARIOS is not None:
        keep = (min(MAX_TEST_SCENARIOS, len(test_idx)) // 5) * 5
        test_idx = test_idx[:keep]   # sorted, contiguous groups of 5 -> groups stay intact
    print(f"  test scenarios used: {len(test_idx)} (never used in training or validation)")

    print("Loading TEST scenarios from disk ...")
    x_test_flat, u_test_flat = fill_split_buffer(
        MERGED_MAT_PATH, backend, test_idx, T, n_states, n_inputs
    )
    if INPUT_COLUMN_PERM is not None:
        u_test_flat = np.ascontiguousarray(u_test_flat[:, INPUT_COLUMN_PERM])

    print("Loading saved scalers (fit on TRAIN only) and scaling TEST data ...")
    with open(os.path.join(OUTPUT_DIR, "scaler_states.pkl"), "rb") as fh:
        scaler_states = load(fh)
    with open(os.path.join(OUTPUT_DIR, "scaler_inputs.pkl"), "rb") as fh:
        scaler_inputs = load(fh)
    transform_in_place(scaler_states, x_test_flat)
    transform_in_place(scaler_inputs, u_test_flat)
    x_test = x_test_flat.reshape(len(test_idx), T, n_states)
    u_test = u_test_flat.reshape(len(test_idx), T, n_inputs)

    cgm_min_scaled = scale_single_state(40, "Q1", OUTPUT_DIR)
    cgm_max_scaled = scale_single_state(400, "Q1", OUTPUT_DIR)

    model_path = os.path.join(OUTPUT_DIR, MODEL_FILENAME)
    print(f"Loading trained model from: {model_path}")
    model = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    simulator = ForwardEulerSimulatorPop(model, cgm_min_scaled, cgm_max_scaled, ts=5.0).to(DEVICE)

    # Non-overlapping windows: overlap=0.0 -> hop = SEQ_LEN (61 steps), so windows tile
    # each scenario without sharing any timestep.
    sampler = FramedWindowSampler(
        x_test, u_test, SEQ_LEN, overlap=0.0, batch_size=EVAL_BATCH_SIZE,
        device=DEVICE, seed=SEED,
    )
    all_pairs = sampler.pairs
    n_windows = len(all_pairs)
    print(f"Total non-overlapping 5-hour test windows: {n_windows}")

    sim_chunks = {k: [] for k in MARGINS}
    act_chunks = {k: [] for k in MARGINS}
    n_skipped = 0

    with torch.no_grad():
        for start in range(0, n_windows, EVAL_BATCH_SIZE):
            pair_batch = all_pairs[start:start + EVAL_BATCH_SIZE]
            x0, u_batch, x_true = sampler.batch_from_pairs(pair_batch)
            x_sim = simulator(x0, u_batch)   # (m, q, 10), scaled

            y_sim = scale_inverse_Q1(x_sim[:, :, [0]].cpu().numpy(), OUTPUT_DIR)[..., 0].T    # (q, m)
            y_true = scale_inverse_Q1(x_true[:, :, [0]].cpu().numpy(), OUTPUT_DIR)[..., 0].T  # (q, m)

            valid = np.isfinite(y_sim).all(axis=1) & np.isfinite(y_true).all(axis=1)
            n_skipped += int((~valid).sum())
            if not valid.any():
                continue
            y_sim, y_true = y_sim[valid], y_true[valid]

            sim_out = window_outcomes_batch(y_sim)
            act_out = window_outcomes_batch(y_true)
            for k in MARGINS:
                sim_chunks[k].append(sim_out[k])
                act_chunks[k].append(act_out[k])

            if (start // EVAL_BATCH_SIZE) % 50 == 0:
                print(f"  processed {min(start + EVAL_BATCH_SIZE, n_windows)}/{n_windows} windows")

    n_used = n_windows - n_skipped
    if n_used < 2:
        raise RuntimeError("Fewer than 2 valid windows - simulation produced NaN/inf. "
                           "Check the trained model.")

    print("\n" + "=" * 90)
    print(f"{'Metric':8s} | {'Actual (ODE sim)':22s} | {'NN-based Population-level':26s} | {'p-value':9s} | Equivalent?")
    print("-" * 90)
    for k in ["TIR", "TAR", "TBR", "LBGI", "HBGI", "MG"]:
        sim_arr = np.concatenate(sim_chunks[k])
        act_arr = np.concatenate(act_chunks[k])
        mean_sim, sd_sim, mean_act, sd_act, p, equiv = paired_tost(sim_arr, act_arr, MARGINS[k])
        unit = "%" if k in ("TIR", "TAR", "TBR") else ("mg/dL" if k == "MG" else "")
        print(f"{k:8s} | {mean_act:6.1f} +/- {sd_act:5.1f} {unit:5s} | "
              f"{mean_sim:6.1f} +/- {sd_sim:5.1f} {unit:5s} | "
              f"{p:9.4f} | {'YES' if equiv else 'no'}")
    print("=" * 90)
    print(f"\nEvaluated {n_used} windows from {len(test_idx)} held-out test scenarios "
          f"({n_skipped} skipped for NaN/inf).")
    print("Equivalence concluded only if p-value < 0.05 (both one-sided tests significant).")


if __name__ == "__main__":
    main()
