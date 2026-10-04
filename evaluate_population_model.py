"""
eval_population_model.py  (fixed version)

Evaluates the trained population NN model on the held-out simulated DP_test split
using the paper's 5-hour glucose outcome metrics and TOST methodology.

Run after train_population_model.py (same folder, same OUTPUT_DIR).
"""

import os
from pickle import load
import numpy as np
import torch
import scipy.stats as st

from t1dsim_ai.population_model import CGMOHSUSimStateSpaceModel_V2
from t1dsim_ai.options import n_neurons_pop
from t1dsim_ai.utils.preprocess import scale_inverse_Q1

from train_population_model import (
    get_dataset_dims,
    get_splits,
    fill_split_buffer,
    transform_in_place,
    FramedWindowSampler,
    ForwardEulerSimulatorPop,
    MERGED_MAT_PATH,
    OUTPUT_DIR,
    MODEL_FILENAME,
    SEED,
    SEQ_LEN,
    INPUT_COLUMN_PERM,
)

# ==============================================================================
# CONFIG
# ==============================================================================
EVAL_BATCH_SIZE = 256
MAX_TEST_SCENARIOS = None   # Evaluate on full TEST split
# [UNRESOLVED] Final population test window overlap/stride.
# Zero overlap means adjacent windows step by 60 intervals (sharing the boundary state).
TEST_OVERLAP = 0.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TIR_LOW, TIR_HIGH = 70.0, 180.0
MARGINS = {"TIR": 5.0, "TAR": 5.0, "TBR": 1.0, "LBGI": 0.5, "HBGI": 1.0, "MG": 10.0}
ALPHA = 0.05

# ==============================================================================
# Clinical outcome metrics
# ==============================================================================
def window_outcomes_batch(bg):
    tir = 100.0 * np.mean((bg >= TIR_LOW) & (bg <= TIR_HIGH), axis=1)
    tar = 100.0 * np.mean(bg > TIR_HIGH, axis=1)
    tbr = 100.0 * np.mean(bg < TIR_LOW, axis=1)

    bgc = np.clip(bg, 1.0, None)
    f_bg = 1.509 * (np.log(bgc) ** 1.084 - 5.381)
    r_bg = 10.0 * f_bg ** 2
    neg = f_bg < 0
    pos = ~neg
    n = bg.shape[1]
    lbgi = (r_bg * neg).sum(axis=1) / n
    hbgi = (r_bg * pos).sum(axis=1) / n

    return {"TIR": tir, "TAR": tar, "TBR": tbr, "LBGI": lbgi, "HBGI": hbgi, "MG": bg.mean(axis=1)}

def paired_tost(sim, actual, margin, alpha=ALPHA):
    diff = sim - actual
    n = len(diff)
    if n < 2:
        return (float('nan'), float('nan'), float('nan'), float('nan'), 1.0, False)
        
    mean_diff = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    
    if se == 0:
        p_value = 0.0 if abs(mean_diff) < margin else 1.0
        return (sim.mean(), sim.std(ddof=1) if n >= 2 else 0.0, 
                actual.mean(), actual.std(ddof=1) if n >= 2 else 0.0,
                p_value, p_value < alpha)

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
    n_scenarios, T, n_states, n_inputs = get_dataset_dims(MERGED_MAT_PATH)

    print("Extracting TEST split from merged dataset...")
    _, _, test_idx = get_splits(MERGED_MAT_PATH)
    if MAX_TEST_SCENARIOS is not None:
        test_idx = test_idx[:MAX_TEST_SCENARIOS]
    print(f"  test scenarios to evaluate: {len(test_idx)}")

    print("Loading TEST scenarios from disk ...")
    x_test_flat, u_test_flat = fill_split_buffer(MERGED_MAT_PATH, test_idx, T, n_states, n_inputs)
    if INPUT_COLUMN_PERM is not None:
        u_test_flat = np.ascontiguousarray(u_test_flat[:, INPUT_COLUMN_PERM])

    print("Loading saved scalers and scaling TEST data ...")
    with open(os.path.join(OUTPUT_DIR, "scaler_states.pkl"), "rb") as fh:
        scaler_states = load(fh)
    with open(os.path.join(OUTPUT_DIR, "scaler_inputs.pkl"), "rb") as fh:
        scaler_inputs = load(fh)
    
    transform_in_place(scaler_states, x_test_flat)
    transform_in_place(scaler_inputs, u_test_flat)
    
    x_test = x_test_flat.reshape(len(test_idx), T, n_states)
    u_test = u_test_flat.reshape(len(test_idx), T, n_inputs)

    model_path = os.path.join(OUTPUT_DIR, MODEL_FILENAME)
    print(f"Loading trained model from: {model_path}")
    model = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Note: no CGM clamp in evaluation per strict adherence.
    simulator = ForwardEulerSimulatorPop(model, ts=5.0).to(DEVICE)

    if TEST_OVERLAP is None:
        raise ValueError("TEST_OVERLAP must be configured manually before evaluation.")

    sampler = FramedWindowSampler(
        x_test, u_test, SEQ_LEN, overlap=TEST_OVERLAP, batch_size=EVAL_BATCH_SIZE,
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
            x_sim = simulator(x0, u_batch)

            y_sim = scale_inverse_Q1(x_sim[:, :, [0]].cpu().numpy(), OUTPUT_DIR)[..., 0].T    # (q, m)
            y_true = scale_inverse_Q1(x_true[:, :, [0]].cpu().numpy(), OUTPUT_DIR)[..., 0].T  # (q, m)

            if not np.isfinite(y_sim).all():
                raise RuntimeError(
                    "Non-finite population-model output detected during final test evaluation."
                )

            if not np.isfinite(y_true).all():
                raise RuntimeError(
                    "Non-finite ground-truth values detected in final test evaluation."
                )

            sim_out = window_outcomes_batch(y_sim)
            act_out = window_outcomes_batch(y_true)
            
            for k in MARGINS:
                sim_chunks[k].append(sim_out[k])
                act_chunks[k].append(act_out[k])

    n_used = n_windows - n_skipped

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

if __name__ == "__main__":
    main()
