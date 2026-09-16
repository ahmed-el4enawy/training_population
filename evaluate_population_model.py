"""
eval_population_model.py

Evaluates the trained population-level NN state-space model
(CGMOHSUSimStateSpaceModel_V2 / T1DSim_NN^P) against the held-out TEST split
of the simulated dataset, reproducing the "NN-based Population-level" vs.
"ODE-based Population-level (Control)" comparison in Table 3 of:

    Roquemen-Echeverri et al., "A Physiologically-Constrained Neural Network
    Digital Twin Framework for Replicating Glucose Dynamics in Type 1
    Diabetes" (arXiv:2508.05705)

WHAT THIS DOES
--------------
1. Re-derives the exact same train/val/TEST scenario split used by
   train_population_model.py (same seed, same group_split), so this script
   evaluates on scenarios the population model never saw during training or
   validation.
2. Streams those TEST scenarios off disk (bounded memory, same approach as
   the training script), and scales them with the *already-fitted* scalers
   saved by training (scaler_states.pkl / scaler_inputs.pkl) -- it does NOT
   refit scalers on test data.
3. Splits every scenario into non-overlapping 5-hour windows (61 steps at
   Ts=5min, matching SEQ_LEN in the training script and the paper's
   "sequences of size M=5 hours with no overlap" evaluation convention,
   Section 2.2.3 / 2.6.2).
4. Runs the trained model forward with Euler integration (population-only,
   no individual-level term) starting from each window's REAL initial
   state, producing a simulated Q1 (mg/dL) trace for every window.
5. Computes, per 5-hour window, for both the simulated trace and the actual
   (ODE-generated ground truth) trace:
       - TIR   (% time 70-180 mg/dL)
       - TAR   (% time >180 mg/dL)
       - TBR   (% time <70 mg/dL)
       - LBGI, HBGI (Kovatchev risk indices)
       - MG    (mean glucose, mg/dL)
6. Runs paired two one-sided t-tests (TOST) comparing simulated vs. actual
   per-window outcomes, using the paper's equivalence margins:
       TIR=5%, TAR=5%, TBR=1%, HBGI=1, LBGI=0.5, MG=10 mg/dL
   at alpha=0.05, and prints a results table in the same layout as Table 3's
   "ODE-based Population-level" / "NN-based Population-level" columns
   (mean +/- SD, larger of the two one-sided p-values).

This script does NOT require Gurobi, and does NOT do individual-level
(digital twin) simulation -- it only reproduces the population-level rows.
If you want the "NN-based Digital Twins" row, you need the trained
individual-level model(s) T1DSim_NN^I plus T1DEXI CGM/HR/sleep data, which
is a separate script.

Requires: torch, numpy, scipy, scikit-learn, h5py and/or scipy.io, and the
t1dsim_ai package (same environment used for training).
"""

import os
import numpy as np
import torch
import scipy.stats as st

from t1dsim_ai.population_model import CGMOHSUSimStateSpaceModel_V2
from t1dsim_ai.options import n_neurons_pop
from t1dsim_ai.utils.preprocess import scale_inverse_Q1

# Re-use the exact data-loading / splitting machinery from the training
# script so the TEST split here is guaranteed identical to the one that
# training never saw. Point this at wherever you saved that file.
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
)

# ==============================================================================
# CONFIG
# ==============================================================================
EVAL_BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clinical thresholds (mg/dL)
TIR_LOW, TIR_HIGH = 70.0, 180.0

# Paper's equivalence margins (Section 2.6.2)
MARGINS = {
    "TIR": 5.0,
    "TAR": 5.0,
    "TBR": 1.0,
    "LBGI": 0.5,
    "HBGI": 1.0,
    "MG": 10.0,
}
ALPHA = 0.05


# ==============================================================================
# Clinical outcome metrics (per 5-hour window)
# ==============================================================================
def kovatchev_risk(bg_mgdl):
    """Returns (LBGI, HBGI) for a 1D array of glucose values in mg/dL,
    using the risk function from Kovatchev et al. (Nature Reviews
    Endocrinology, 2017), as used by the paper (ref [41])."""
    bg = np.clip(bg_mgdl, 1.0, None)  # guard against log(0) on invalid sims
    f_bg = 1.509 * (np.log(bg) ** 1.084 - 5.381)
    r_bg = 10.0 * f_bg**2
    lbgi = r_bg[f_bg < 0].mean() if np.any(f_bg < 0) else 0.0
    hbgi = r_bg[f_bg >= 0].mean() if np.any(f_bg >= 0) else 0.0
    return lbgi, hbgi


def window_outcomes(bg_mgdl):
    """bg_mgdl: 1D array, one 5-hour window's glucose trace (mg/dL).
    Returns dict of TIR/TAR/TBR/LBGI/HBGI/MG for that window."""
    n = len(bg_mgdl)
    tir = 100.0 * np.sum((bg_mgdl >= TIR_LOW) & (bg_mgdl <= TIR_HIGH)) / n
    tar = 100.0 * np.sum(bg_mgdl > TIR_HIGH) / n
    tbr = 100.0 * np.sum(bg_mgdl < TIR_LOW) / n
    lbgi, hbgi = kovatchev_risk(bg_mgdl)
    mg = bg_mgdl.mean()
    return {"TIR": tir, "TAR": tar, "TBR": tbr, "LBGI": lbgi, "HBGI": hbgi, "MG": mg}


# ==============================================================================
# Paired TOST equivalence test
# ==============================================================================
def paired_tost(sim, actual, margin, alpha=ALPHA):
    """Two one-sided paired t-tests (TOST) for equivalence within +/-margin.
    Returns (mean_sim, sd_sim, mean_actual, sd_actual, p_value, equivalent)
    where p_value is the LARGER of the two one-sided p-values (paper
    convention), and equivalent = True iff p_value < alpha."""
    diff = sim - actual
    n = len(diff)
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    se = sd_diff / np.sqrt(n)

    # H0: mean_diff <= -margin  vs  H1: mean_diff > -margin
    t1 = (mean_diff - (-margin)) / se
    p1 = 1 - st.t.cdf(t1, df=n - 1)

    # H0: mean_diff >= margin   vs  H1: mean_diff < margin
    t2 = (mean_diff - margin) / se
    p2 = st.t.cdf(t2, df=n - 1)

    p_value = max(p1, p2)
    equivalent = p_value < alpha

    return (
        sim.mean(), sim.std(ddof=1),
        actual.mean(), actual.std(ddof=1),
        p_value, equivalent,
    )


# ==============================================================================
# Main
# ==============================================================================
def main():
    print(f"Loading dataset dims from: {MERGED_MAT_PATH}")
    backend = _detect_backend(MERGED_MAT_PATH)
    n_scenarios, T, n_states, n_inputs = get_dataset_dims(MERGED_MAT_PATH, backend)

    print("Re-deriving train/val/TEST split (must match training exactly) ...")
    train_idx, val_idx, test_idx = group_split(
        n_scenarios, group_size=5, train_frac=TRAIN_FRACTION, val_frac=VAL_FRACTION, seed=SEED
    )
    print(f"  test scenarios: {len(test_idx)} (never used in training or validation)")

    print(f"Streaming TEST scenarios from disk ...")
    x_test_flat, u_test_flat = fill_split_buffer(
        MERGED_MAT_PATH, backend, test_idx, T, n_states, n_inputs
    )

    print("Loading saved scalers (fit on TRAIN only) and scaling TEST data ...")
    from pickle import load
    scaler_states = load(open(os.path.join(OUTPUT_DIR, "scaler_states.pkl"), "rb"))
    scaler_inputs = load(open(os.path.join(OUTPUT_DIR, "scaler_inputs.pkl"), "rb"))
    transform_in_place(scaler_states, x_test_flat)
    transform_in_place(scaler_inputs, u_test_flat)
    x_test = x_test_flat.reshape(len(test_idx), T, n_states)
    u_test = u_test_flat.reshape(len(test_idx), T, n_inputs)

    from t1dsim_ai.utils.preprocess import scale_single_state
    cgm_min_scaled = scale_single_state(40, "Q1", OUTPUT_DIR)
    cgm_max_scaled = scale_single_state(400, "Q1", OUTPUT_DIR)

    print(f"Loading trained model from: {os.path.join(OUTPUT_DIR, MODEL_FILENAME)}")
    model = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
    model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, MODEL_FILENAME), map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    simulator = ForwardEulerSimulatorPop(model, cgm_min_scaled, cgm_max_scaled, ts=5.0).to(DEVICE)

    # --- Build NON-OVERLAPPING 5-hour windows across every test scenario ---
    # hop = SEQ_LEN - 1 so consecutive windows tile the scenario with no
    # overlap other than sharing one boundary timestep, matching the paper's
    # "no overlap" evaluation convention (Sec. 2.2.3 / 2.6.2).
    sampler = FramedWindowSampler(
        x_test, u_test, SEQ_LEN, overlap=0.0, batch_size=EVAL_BATCH_SIZE,
        device=DEVICE, seed=SEED,
    )
    all_pairs = sampler.pairs
    n_windows = len(all_pairs)
    print(f"Total non-overlapping 5-hour test windows: {n_windows}")

    sim_outcomes = {k: [] for k in MARGINS}
    actual_outcomes = {k: [] for k in MARGINS}

    with torch.no_grad():
        for start in range(0, n_windows, EVAL_BATCH_SIZE):
            pair_batch = all_pairs[start:start + EVAL_BATCH_SIZE]
            x0, u_batch, x_true = sampler.batch_from_pairs(pair_batch)
            x_sim = simulator(x0, u_batch)  # (m, q, 10), scaled

            y_sim_mgdl = scale_inverse_Q1(x_sim[:, :, [0]].cpu().numpy(), OUTPUT_DIR)   # (m, q, 1)
            y_true_mgdl = scale_inverse_Q1(x_true[:, :, [0]].cpu().numpy(), OUTPUT_DIR)  # (m, q, 1)

            y_sim_mgdl = y_sim_mgdl[..., 0].T   # (q, m)
            y_true_mgdl = y_true_mgdl[..., 0].T  # (q, m)

            for w in range(y_sim_mgdl.shape[0]):
                sim_out = window_outcomes(y_sim_mgdl[w])
                act_out = window_outcomes(y_true_mgdl[w])
                for k in MARGINS:
                    sim_outcomes[k].append(sim_out[k])
                    actual_outcomes[k].append(act_out[k])

            if (start // EVAL_BATCH_SIZE) % 20 == 0:
                print(f"  processed {min(start + EVAL_BATCH_SIZE, n_windows)}/{n_windows} windows")

    # --- Report, matching Table 3's layout ---
    print("\n" + "=" * 90)
    print(f"{'Metric':8s} | {'Actual (ODE sim)':22s} | {'NN-based Population-level':26s} | {'p-value':9s} | Equivalent?")
    print("-" * 90)
    results = {}
    for k in ["TIR", "TAR", "TBR", "LBGI", "HBGI", "MG"]:
        sim_arr = np.array(sim_outcomes[k])
        act_arr = np.array(actual_outcomes[k])
        mean_sim, sd_sim, mean_act, sd_act, p, equiv = paired_tost(sim_arr, act_arr, MARGINS[k])
        results[k] = dict(mean_sim=mean_sim, sd_sim=sd_sim, mean_act=mean_act, sd_act=sd_act, p=p, equiv=equiv)
        unit = "%" if k in ("TIR", "TAR", "TBR") else ("mg/dL" if k == "MG" else "")
        print(
            f"{k:8s} | {mean_act:6.1f} +/- {sd_act:5.1f} {unit:5s} | "
            f"{mean_sim:6.1f} +/- {sd_sim:5.1f} {unit:5s} | "
            f"{p:9.4f} | {'YES' if equiv else 'no'}"
        )
    print("=" * 90)
    print(f"\nEvaluated on {n_windows} non-overlapping 5-hour windows from {len(test_idx)} held-out test scenarios.")
    print("Equivalence concluded only if p-value < 0.05 (both one-sided tests significant).")


if __name__ == "__main__":
    main()