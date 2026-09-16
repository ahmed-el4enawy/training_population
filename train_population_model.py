"""
train_population_model.py

Trains the population-level NN state-space model T1DSim_NN^P
(CGMOHSUSimStateSpaceModel_V2) on the merged simulated development dataset
(population_development_dataset_merged.mat), following the methodology in
Section 2.2.2 of:

    Roquemen-Echeverri et al., "A Physiologically-Constrained Neural Network
    Digital Twin Framework for Replicating Glucose Dynamics in Type 1
    Diabetes" (arXiv:2508.05705)

WHAT THIS DOES
--------------
1. Reads dataset_states (N,T,10), dataset_inputs (N,T,2), dataset_glucose
   (N,T) from the merged .mat file -- but STREAMED IN BOUNDED CHUNKS
   straight off disk, never materializing the full 46200-scenario dataset
   in RAM (see MEMORY NOTES below).
2. IMPORTANT CORRECTION: dataset_states[...,0] is Q1 in mmol/kg (the raw
   ODE mass unit written by the MATLAB generator). The rest of this
   codebase (individual_model.py) treats state index 0 ("Q1"/"output_cgm")
   as the CGM value in mg/dL directly (see LIM_INFERIOR/SUPERIOR and
   cgm_min/max, which apply 70/250/40/400 mg/dL thresholds straight to
   this state). So we substitute dataset_glucose (mg/dL) in as channel 0.
3. Splits scenarios into train/val/test (60/20/20) GROUPED by meal pattern:
   the MATLAB generator produces 5 consecutive scenarios (5 initial-glucose
   variants) per meal pattern, so grouping by floor(row_index/5) keeps the
   same meal pattern from appearing in more than one split, matching the
   paper's leakage-avoidance design. Test scenarios are never loaded at all
   since this script doesn't use them.
4. Fits a fresh RobustScaler on TRAIN states/inputs and saves
   scaler_states.pkl / scaler_inputs.pkl to OUTPUT_DIR, in the exact format
   t1dsim_ai.utils.preprocess.scaler()/scale_single_state()/
   scale_inverse_Q1() expect, so the rest of the codebase (individual
   model training, DigitalTwin simulation) can load this population model
   normally afterwards.
5. Trains CGMOHSUSimStateSpaceModel_V2 with Euler-integration simulation
   error minimization (Forgione & Piga), using:
       L_total = L_fit + alpha * L_consistency         (Eq. 5)
       L_fit: MSE on simulated vs real Q1, with a penalty upweighting
              hypo (<70 mg/dL) and hyper (>250 mg/dL) errors  (Eq. 6-7)
       L_consistency: weighted MSE across all 10 states, each with
              a soft non-negativity penalty vs. the training-set minimum
              (Eq. 8-9), except C1 (no ground truth) whose floor is fixed
              to 0 as noted in the paper.
   Batches use TRUE 75% overlap framing, matching individual_model.py's
   Batch class exactly (every overlapping window is visited once per
   epoch, shuffled) -- but at population scale (46,200 scenarios x 2017
   steps) we can't materialize every window's actual data upfront like
   that class does (tens of GB of RAM). Instead FramedWindowSampler
   precomputes only the (scenario_idx, start_idx) index pairs for every
   window (a few hundred MB) and slices the real window data out of the
   raw arrays on demand, batch by batch.
6. Early stopping on validation loss (patience=150 epochs), the same
   heuristic already present -- commented out -- in individual_model.py's
   IndividualModel.fit(), since the paper doesn't state a fixed epoch
   count for population-model training (only 20 epochs as the *search*
   budget during the Section 2.2.3 Bayesian optimization of neuron
   counts, which is a separate procedure already reflected in
   options.py's n_neurons_pop).

MEMORY NOTES (why this version differs from a "naive" loader)
---------------------------------------------------------------
A straightforward loader that does
    dataset_states = mat['dataset_states']          # full (46200,2017,10)
    x_est = dataset_states.astype(float32)          # a second full copy
    x_train = x_est[train_idx]                      # a third, ~2GiB copy
    x_train_scaled = scaler.transform(...)          # a fourth, ~2GiB copy
ends up with several multi-GiB arrays alive in RAM at once (worse if the
.mat file stores float64, which v7.3/HDF5-based MAT files -- the ones
scipy can't read and h5py has to handle -- commonly do: that alone makes
the raw dataset_states array ~7.5 GiB instead of ~3.7 GiB).

This version instead:
  - Gets N/T/n_states/n_inputs from the file's variable *shape* only
    (scipy.io.whosmat, or the h5py Dataset.shape) without reading any
    array data, so train/val/test can be split before anything is loaded.
  - Streams ONLY the train and val scenario indices off disk in bounded
    chunks (CHUNK_SCENARIOS at a time), casting to float32 and doing the
    Q1 substitution per chunk, writing directly into a single preallocated
    flat (n_split*T, n_features) float32 buffer per split. Test scenarios
    are never read from disk at all.
  - Fits RobustScaler on that buffer, then calls scaler.transform() in
    row-chunks and writes the result BACK INTO THE SAME BUFFER, so there
    is never a second full-split-sized "scaled" copy sitting next to the
    "raw" one.
  - The (n_split*T, n_features) buffer is reshaped to (n_split,T,n_features)
    for use by FramedWindowSampler; since the buffer is contiguous this
    reshape is a view, not a copy.

Requires: torch, numpy, scikit-learn, h5py and/or scipy, and the t1dsim_ai
package installed (pip install -e . from the repo, or the package already
present in your environment, since population_model.py / options.py /
utils/preprocess.py are imported from it).
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pickle import dump

try:
    import scipy.io as sio
except ImportError:
    sio = None

try:
    import h5py
except ImportError:
    h5py = None

from t1dsim_ai.population_model import CGMOHSUSimStateSpaceModel_V2
from t1dsim_ai.options import n_neurons_pop
from t1dsim_ai.utils.preprocess import scale_single_state, scale_inverse_Q1

# ==============================================================================
# CONFIG - EDIT THESE
# ==============================================================================
MERGED_MAT_PATH = r"C:\Users\pc\Downloads\population\population_development_dataset_merged.mat"
OUTPUT_DIR = "models/PopulationModel/"          # scaler_*.pkl + trained model saved here
MODEL_FILENAME = "population_model_trained.pt"
WARM_START_CHECKPOINT = None                    # e.g. "models/PopulationModel/population_model_05022024_epoch_15.pt" to continue training, or None to train from scratch

SEQ_LEN = 61            # 5 hours at Ts=5min (60 steps + 1 initial), matches individual_model.py convention
OVERLAP = 0.75           # 75% overlap between consecutive training sequences (paper Section 2.2.2) - now
                         # used for TRUE window framing (see FramedWindowSampler), not just an iteration count
BATCH_SIZE = 128
MAX_EPOCHS = 150         # user-chosen budget (~40 min/epoch on this GPU at current settings -> ~66-67 hrs / ~2.8
                         # days if it runs the full amount). The paper doesn't state a fixed epoch count for
                         # population-model training (only 20 epochs as the *search* budget during Bayesian
                         # optimization of neuron counts, a separate procedure).
PATIENCE = 20            # epochs without val-loss improvement before stopping early (must be < MAX_EPOCHS to
                         # ever actually trigger - the original 150 was higher than MAX_EPOCHS and so could
                         # never fire). Lower = willing to stop sooner if it plateaus; raise toward MAX_EPOCHS
                         # if you'd rather guarantee it uses the full 100-epoch budget regardless of plateaus.

# --- epoch-size cap (new) ---
# With true 75% overlap this dataset produces millions of windows/epoch (3.6M for train
# at these settings). On a small/underpowered GPU (or CPU), the Euler simulator's
# sequential per-timestep loop is overhead-bound, not compute-bound, so visiting every
# single window every single epoch can make one epoch take hours while the GPU sits
# mostly idle between kernel launches. Setting these to an integer instead of None
# makes each epoch sample that many RANDOM batches (without replacement, reshuffled
# once exhausted) instead of the full window set - over many epochs (you have up to
# MAX_EPOCHS with early stopping) the dataset still gets covered many times over
# statistically, at a fraction of the wall-clock cost per epoch. Set to None to restore
# the literal "every window, once per epoch" behavior.
MAX_TRAIN_BATCHES_PER_EPOCH = 3000   # None = use every training window every epoch (original behavior)
MAX_VAL_BATCHES_PER_EPOCH = 1000     # None = use every validation window every epoch (original behavior)
LR = 1e-3
LR_DECAY_PER_EPOCH = np.exp(-0.1)   # paper: lr scheduled to be reduced by e^-0.1 every epoch
WEIGHT_DECAY = 0.0        # not specified for the population model (unlike 1e-5 for the individual model)
ALPHA = 0.7               # Eq. 5: weight on consistency loss
BETA = 0.08               # Eq. 9: weight on soft non-negativity penalty
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2       # remainder (0.2) is held out as test, unused by this script (and never loaded)
SEED = 0

# --- memory-management knobs (new) ---
CHUNK_SCENARIOS = 2000    # how many raw scenarios to read from disk / cast to float32 at a time while
                          # streaming a split into its flat buffer. Lower this if you still run out of RAM
                          # (e.g. to 500); it only affects peak memory and speed, not results.
CHUNK_ROWS = 500_000      # how many rows (time steps) at a time to run through scaler.transform() when
                          # scaling a split's flat buffer in place. Lower this if RAM is still tight.

# State order used throughout this codebase: [Q1, Q2, S1, S2, I, X1, X2, X3, C2, C1]
STATE_ORDER = ["Q1", "Q2", "S1", "S2", "I", "X1", "X2", "X3", "C2", "C1"]
C1_INDEX = STATE_ORDER.index("C1")  # no ground truth for C1 -> floor fixed to 0, per paper

# Per-state consistency weights (Eq. 8), from Section 2.2.3:
# w_i = 0.083bar for S1,S2,I,X1,X2,X3,C2 (=1/12 each) ; w_Q1 = 0.2083bar (=5/24) ;
# w_Q2 = 0.16bar (=1/6) ; w_C1 = 0.0416bar (=1/24). These are repeating decimals in the
# paper, constrained to sum to exactly 1 (Eq. 8: sum_i w_i = 1) - using exact fractions
# here instead of 3-decimal-place rounding (the previous 0.2083/0.16/0.083/0.0416 summed
# to 0.9909, not 1).
STATE_WEIGHTS = np.array(
    [5/24, 1/6, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/24],
    dtype=np.float32,
)  # order matches STATE_ORDER above: [Q1, Q2, S1, S2, I, X1, X2, X3, C2, C1]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# Data loading - streamed, chunked, memory-bounded (see MEMORY NOTES above)
# ==============================================================================
def _load_var_scipy(path, key):
    d = sio.loadmat(path, variable_names=[key], simplify_cells=True)
    return d[key]


def _detect_backend(path):
    """Legacy (<v7.3) MAT files are read with scipy; v7.3/HDF5-based ones
    (typical once a variable exceeds ~2GB, which is exactly this dataset's
    situation) can only be read with h5py. We probe with scipy.io.whosmat
    first (reads only the header, not the data) and fall back to h5py."""
    if sio is not None:
        try:
            sio.whosmat(path)
            return "scipy"
        except Exception:
            pass
    if h5py is not None:
        return "h5py"
    raise RuntimeError("Need scipy and/or h5py installed to read this .mat file")


def get_dataset_dims(path, backend):
    """Returns (n_scenarios, T, n_states, n_inputs) by reading only variable
    shapes/headers, never the underlying array data."""
    if backend == "scipy":
        shapes = {name: shape for name, shape, _ in sio.whosmat(path)}
        n_scenarios, T, n_states = shapes["dataset_states"]
        n_inputs = shapes["dataset_inputs"][-1]
        return n_scenarios, T, n_states, n_inputs
    else:
        with h5py.File(path, "r") as f:
            # v7.3/HDF5 stores MATLAB arrays axis-reversed vs. how numpy will
            # eventually see them (dataset_states ends up (N,T,10), so on disk
            # it's (10,T,N); dataset_inputs ends up (N,T,2), on disk (2,T,N)).
            n_states, T, n_scenarios = f["dataset_states"].shape
            n_inputs = f["dataset_inputs"].shape[0]
        return n_scenarios, T, n_states, n_inputs


def _iter_raw_chunks_scipy(path, chunk_size):
    # scipy can't do partial reads of a MAT variable, so this backend still
    # pays for one full load of each variable (only used for legacy, non-v7.3
    # files, which are the ones scipy can parse at all) -- but note it loads
    # each variable exactly ONCE here, then hands out float32 chunk views,
    # instead of the original script's pattern of building a full float32
    # copy AND THEN taking further full-split-sized copies out of it.
    dataset_states = _load_var_scipy(path, "dataset_states")
    dataset_inputs = _load_var_scipy(path, "dataset_inputs")
    dataset_glucose = _load_var_scipy(path, "dataset_glucose")
    N = dataset_states.shape[0]
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        states_chunk = dataset_states[start:end].astype(np.float32)
        inputs_chunk = dataset_inputs[start:end].astype(np.float32)
        glucose_chunk = dataset_glucose[start:end].astype(np.float32)
        states_chunk[..., 0] = glucose_chunk  # Q1 mg/dL correction
        yield start, end, states_chunk, inputs_chunk


def _iter_raw_chunks_h5py(path, chunk_size):
    # Contiguous-range slicing along the on-disk scenario axis -> plain HDF5
    # hyperslab reads (fast, no "fancy indexing" involved), and each chunk is
    # small and short-lived (default 2000 scenarios ~= 161 MB for states).
    with h5py.File(path, "r") as f:
        ds_states = f["dataset_states"]    # on-disk shape (10, T, N)
        ds_inputs = f["dataset_inputs"]    # on-disk shape (2, T, N)
        ds_glucose = f["dataset_glucose"]  # on-disk shape (T, N)
        N = ds_states.shape[-1]
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            states_chunk = np.transpose(ds_states[:, :, start:end], (2, 1, 0)).astype(np.float32)   # (c,T,10)
            inputs_chunk = np.transpose(ds_inputs[:, :, start:end], (2, 1, 0)).astype(np.float32)    # (c,T,2)
            glucose_chunk = np.transpose(ds_glucose[:, start:end], (1, 0)).astype(np.float32)        # (c,T)
            states_chunk[..., 0] = glucose_chunk  # Q1 mg/dL correction
            yield start, end, states_chunk, inputs_chunk


def fill_split_buffer(path, backend, idx, T, n_states, n_inputs, chunk_size=CHUNK_SCENARIOS):
    """Streams only the requested raw scenario indices `idx` off disk, in
    bounded-size chunks, and writes them directly into preallocated flat
    (len(idx)*T, n_states) / (len(idx)*T, n_inputs) float32 buffers.
    At no point does a second full-split-sized copy exist alongside these.
    `idx` must be sorted ascending (group_split already returns it that way).
    """
    idx = np.asarray(idx)
    assert np.all(np.diff(idx) >= 0), "idx must be sorted ascending"
    pos_of = {raw_i: split_pos for split_pos, raw_i in enumerate(idx)}

    x_flat = np.empty((len(idx) * T, n_states), dtype=np.float32)
    u_flat = np.empty((len(idx) * T, n_inputs), dtype=np.float32)

    chunk_iter = (
        _iter_raw_chunks_scipy(path, chunk_size)
        if backend == "scipy"
        else _iter_raw_chunks_h5py(path, chunk_size)
    )

    n_found = 0
    for start, end, states_chunk, inputs_chunk in chunk_iter:
        in_range = idx[(idx >= start) & (idx < end)]
        if len(in_range) == 0:
            continue
        for raw_i in in_range:
            local = raw_i - start
            split_pos = pos_of[raw_i]
            r0, r1 = split_pos * T, (split_pos + 1) * T
            x_flat[r0:r1] = states_chunk[local]
            u_flat[r0:r1] = inputs_chunk[local]
            n_found += 1
        if n_found == len(idx):
            break  # every scenario in this split has been read; skip the rest of the file
                   # (relevant when val_idx's last raw index is well before N, and matters a
                   # lot for the scipy backend where remaining chunks are still cheap since
                   # the full arrays are already loaded, but keeps h5py reads minimal too)

    return x_flat, u_flat


def transform_in_place(scaler, flat_array, chunk_rows=CHUNK_ROWS):
    """Applies scaler.transform() over `flat_array` in row-chunks and writes
    the result back into the SAME buffer, so scaling a split never requires a
    second full-split-sized array the way `scaler.transform(x).reshape(...)`
    (allocating a fresh output array) did in the original script."""
    n = flat_array.shape[0]
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        flat_array[start:end] = scaler.transform(flat_array[start:end]).astype(np.float32)


def group_split(n_scenarios, group_size=5, train_frac=0.6, val_frac=0.2, seed=0):
    """Splits scenario indices into train/val/test, keeping every
    `group_size` consecutive scenarios (one meal pattern's initial-glucose
    variants, per the MATLAB generator) together in the same split."""
    assert n_scenarios % group_size == 0, (
        f"n_scenarios ({n_scenarios}) isn't divisible by group_size "
        f"({group_size}) - meal-pattern grouping assumption doesn't hold, "
        "check how your parts were generated/merged."
    )
    n_groups = n_scenarios // group_size
    rng = np.random.RandomState(seed)
    group_order = rng.permutation(n_groups)

    n_train_groups = int(round(n_groups * train_frac))
    n_val_groups = int(round(n_groups * val_frac))

    train_groups = group_order[:n_train_groups]
    val_groups = group_order[n_train_groups : n_train_groups + n_val_groups]
    test_groups = group_order[n_train_groups + n_val_groups :]

    def groups_to_indices(groups):
        idx = (groups[:, None] * group_size + np.arange(group_size)[None, :]).reshape(-1)
        return np.sort(idx)

    return (
        groups_to_indices(train_groups),
        groups_to_indices(val_groups),
        groups_to_indices(test_groups),
    )


# ==============================================================================
# Framed batch sampler - TRUE 75% overlap, matching the paper and
# individual_model.py's Batch class exactly, but storing only the
# (scenario_idx, start_idx) PAIR for every window instead of the
# materialized window itself. At population scale this index table is a
# few hundred MB (num_windows x 2 int64) instead of tens of GB, while every
# epoch still walks over every overlapping window exactly once (shuffled),
# same as individual_model.py's Batch class does per-subject.
# ==============================================================================
class FramedWindowSampler:
    def __init__(self, x_est, u_fit, seq_len, overlap, batch_size, device, seed=0):
        self.x_est = x_est  # (N, T, 10) float32, already SCALED
        self.u_fit = u_fit  # (N, T, 2) float32, already SCALED
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.N, self.T, self.n_states = x_est.shape

        # hop = int((1 - overlap) * seq_len), same formula as
        # individual_model.py's Batch.__init__ (there called self.overlap,
        # confusingly - it's actually the hop length, not the overlap
        # fraction)
        self.hop = max(1, int((1 - overlap) * seq_len))
        starts = np.arange(0, self.T - self.seq_len + 1, self.hop)

        scenario_grid, start_grid = np.meshgrid(
            np.arange(self.N), starts, indexing="ij"
        )
        # (num_windows_total, 2): column 0 = scenario idx, column 1 = start idx
        self.pairs = np.stack([scenario_grid.ravel(), start_grid.ravel()], axis=1)

        self.rng = np.random.RandomState(seed)
        self._reshuffle()

    def _reshuffle(self):
        self.epoch_order = self.rng.permutation(len(self.pairs))
        self.pos = 0

    def n_batches_per_epoch(self):
        return len(self.pairs) // self.batch_size

    def next_batch(self):
        if self.pos + self.batch_size > len(self.pairs):
            self._reshuffle()

        chosen = self.epoch_order[self.pos : self.pos + self.batch_size]
        self.pos += self.batch_size

        return self.batch_from_pairs(self.pairs[chosen])

    def get_fixed_subset(self, n_batches):
        """Selects a FIXED subset of n_batches*batch_size windows, chosen once
        via this sampler's own RNG, and returns them as a list of (batch_size, 2)
        index arrays - one per batch. Intended for VALIDATION: next_batch() keeps
        advancing persistent internal state (self.pos) call after call, so when
        only a capped number of batches are used per epoch (MAX_VAL_BATCHES_PER_EPOCH),
        successive epochs land on different, non-reshuffled slices of the window
        set - meaning val_loss ends up measuring "which random slice did we land
        on" as much as "did the model improve," which corrupts early stopping and
        best-model selection. Calling this ONCE before training and replaying the
        same returned batches every epoch (via batch_from_pairs) fixes that: every
        epoch evaluates the exact same held-out windows, so val_loss is comparable
        across epochs."""
        n_needed = min(n_batches * self.batch_size, len(self.pairs))
        chosen = self.rng.permutation(len(self.pairs))[:n_needed]
        return [
            self.pairs[chosen[start : start + self.batch_size]]
            for start in range(0, len(chosen), self.batch_size)
        ]

    def batch_from_pairs(self, pair_batch):
        """Builds a (x0, u_batch, x_true) batch from an explicit (batch_size, 2)
        array of (scenario_idx, start_idx) pairs - e.g. one of the fixed batches
        returned by get_fixed_subset(), or a batch chosen by next_batch()."""
        scenario_idx = pair_batch[:, 0]
        start_idx = pair_batch[:, 1]
        offsets = np.arange(self.seq_len)
        idx_range = start_idx[:, None] + offsets[None, :]  # (B, seq_len)

        x_batch = self.x_est[scenario_idx[:, None], idx_range]  # (B, seq_len, 10)
        u_batch = self.u_fit[scenario_idx[:, None], idx_range]  # (B, seq_len, 2)

        # ForwardEulerSimulator expects (m=seq_len, q=batch, n_x) ordering
        x_batch = np.transpose(x_batch, (1, 0, 2))
        u_batch = np.transpose(u_batch, (1, 0, 2))

        x0 = x_batch[0]  # (B, 10) - real initial state at window start

        return (
            torch.tensor(x0, dtype=torch.float32, device=self.device),
            torch.tensor(u_batch, dtype=torch.float32, device=self.device),
            torch.tensor(x_batch, dtype=torch.float32, device=self.device),
        )


# ==============================================================================
# Simulator (population-only; no individual/u_ind term)
# ==============================================================================
class ForwardEulerSimulatorPop(nn.Module):
    def __init__(self, ss_pop_model, cgm_min, cgm_max, ts=5.0):
        super().__init__()
        self.ss_pop_model = ss_pop_model
        self.ts = ts
        self.cgm_min = cgm_min
        self.cgm_max = cgm_max

    def adjust_cgm(self, x):
        x = torch.clamp(x, min=self.cgm_min, max=self.cgm_max)
        return x

    def forward(self, x0_batch, u_batch):
        X_sim_list = []
        x_step = x0_batch
        for step in range(u_batch.shape[0]):
            u_step = u_batch[step]
            X_sim_list.append(x_step)
            dx = self.ss_pop_model(x_step, u_step)
            x_step = x_step + self.ts * dx
            x_step = torch.cat(
                [self.adjust_cgm(x_step[:, [0]]), x_step[:, 1:]], dim=1
            )
        return torch.stack(X_sim_list, 0)  # (m, q, n_x)


# ==============================================================================
# Loss (Eq. 5-9)
# ==============================================================================
class PopulationLoss:
    def __init__(self, state_weights, state_min, c1_index, alpha=0.7, beta=0.08):
        self.w = torch.tensor(state_weights, dtype=torch.float32)
        self.state_min = torch.tensor(state_min, dtype=torch.float32)
        self.c1_index = c1_index
        self.alpha = alpha
        self.beta = beta

    def to(self, device):
        self.w = self.w.to(device)
        self.state_min = self.state_min.to(device)
        return self

    @staticmethod
    def fit_penalty(y_sim, y_true, lim_inferior, lim_superior):
        # Eq. 7: 2x under-prediction penalty below 70 mg/dL, 6x over-prediction
        # penalty above 250 mg/dL (population-model penalty values), else 1x.
        penalty = torch.ones_like(y_true)
        penalty[torch.logical_and(y_true <= lim_inferior, y_sim > y_true)] = 2.0
        penalty[torch.logical_and(y_true >= lim_superior, y_sim < y_true)] = 6.0
        return penalty

    def __call__(self, x_sim, x_true, lim_inferior_scaled, lim_superior_scaled):
        # --- Fit loss on Q1 (Eq. 6) ---
        y_sim = x_sim[1:, :, [0]]
        y_true = x_true[1:, :, [0]]
        penalty = self.fit_penalty(y_sim, y_true, lim_inferior_scaled, lim_superior_scaled)
        L_fit = torch.mean((y_sim - y_true) ** 2 * penalty)

        # --- Consistency loss across all 10 states (Eq. 8-9) ---
        err = x_sim[1:] - x_true[1:]  # (m-1, q, 10)
        mse_per_state = torch.mean(err ** 2, dim=(0, 1))  # (10,)

        floor = self.state_min.clone()
        floor[self.c1_index] = 0.0  # no ground truth for C1, floor fixed to 0 (paper note)
        soft_constraint = torch.clamp(-(x_sim[1:] - floor), min=0.0)
        soft_penalty_per_state = torch.sum(soft_constraint, dim=(0, 1))  # (10,)

        per_state_loss = mse_per_state + self.beta * soft_penalty_per_state
        L_consistency = torch.sum(self.w * per_state_loss)

        L_total = L_fit + self.alpha * L_consistency
        return L_total, L_fit.item(), L_consistency.item()


# ==============================================================================
# Main training routine
# ==============================================================================
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Inspecting merged dataset at: {MERGED_MAT_PATH}")
    t0 = time.time()
    backend = _detect_backend(MERGED_MAT_PATH)
    n_scenarios, T, n_states, n_inputs = get_dataset_dims(MERGED_MAT_PATH, backend)
    print(f"  backend: {backend} | shape: ({n_scenarios}, {T}, {n_states}) states, "
          f"({n_scenarios}, {T}, {n_inputs}) inputs [header-only read, {time.time()-t0:.2f}s]")

    print("Splitting scenarios into train/val/test (grouped by meal pattern) ...")
    train_idx, val_idx, test_idx = group_split(
        n_scenarios, group_size=5, train_frac=TRAIN_FRACTION, val_frac=VAL_FRACTION, seed=SEED
    )
    print(f"  train: {len(train_idx)} scenarios | val: {len(val_idx)} | "
          f"test: {len(test_idx)} (unused here - never read from disk)")

    print(f"Streaming TRAIN scenarios from disk in chunks of {CHUNK_SCENARIOS} ...")
    t0 = time.time()
    x_train_flat, u_train_flat = fill_split_buffer(
        MERGED_MAT_PATH, backend, train_idx, T, n_states, n_inputs
    )
    print(f"  done in {time.time()-t0:.1f}s | x_train_flat {x_train_flat.nbytes/1e9:.2f} GB, "
          f"u_train_flat {u_train_flat.nbytes/1e9:.2f} GB")

    print("Fitting RobustScaler on training data ...")
    from sklearn.preprocessing import RobustScaler

    scaler_states = RobustScaler()
    scaler_inputs = RobustScaler()
    scaler_states.fit(x_train_flat)
    scaler_inputs.fit(u_train_flat)

    dump(scaler_states, open(os.path.join(OUTPUT_DIR, "scaler_states.pkl"), "wb"))
    dump(scaler_inputs, open(os.path.join(OUTPUT_DIR, "scaler_inputs.pkl"), "wb"))
    print(f"  saved scaler_states.pkl / scaler_inputs.pkl to {OUTPUT_DIR}")

    print("Scaling TRAIN split in place (no second full-size copy) ...")
    transform_in_place(scaler_states, x_train_flat)
    transform_in_place(scaler_inputs, u_train_flat)
    x_train = x_train_flat.reshape(len(train_idx), T, n_states)  # view, not a copy
    u_train = u_train_flat.reshape(len(train_idx), T, n_inputs)  # view, not a copy

    print(f"Streaming VAL scenarios from disk in chunks of {CHUNK_SCENARIOS} ...")
    t0 = time.time()
    x_val_flat, u_val_flat = fill_split_buffer(
        MERGED_MAT_PATH, backend, val_idx, T, n_states, n_inputs
    )
    print(f"  done in {time.time()-t0:.1f}s | x_val_flat {x_val_flat.nbytes/1e9:.2f} GB, "
          f"u_val_flat {u_val_flat.nbytes/1e9:.2f} GB")

    print("Scaling VAL split in place ...")
    transform_in_place(scaler_states, x_val_flat)
    transform_in_place(scaler_inputs, u_val_flat)
    x_val = x_val_flat.reshape(len(val_idx), T, n_states)
    u_val = u_val_flat.reshape(len(val_idx), T, n_inputs)

    # --- Per-state min (scaled) on TRAIN, for the Eq. 9 soft constraint ---
    state_min = x_train.reshape(-1, n_states).min(axis=0)

    # --- Clinical thresholds (70/250 mg/dL) in scaled space, for the fit penalty ---
    lim_inferior_scaled = scale_single_state(70, "Q1", OUTPUT_DIR)
    lim_superior_scaled = scale_single_state(250, "Q1", OUTPUT_DIR)
    cgm_min_scaled = scale_single_state(40, "Q1", OUTPUT_DIR)
    cgm_max_scaled = scale_single_state(400, "Q1", OUTPUT_DIR)

    # --- Model, simulator, loss, optimizer ---
    print(f"Building model on device: {DEVICE}")
    model = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
    if WARM_START_CHECKPOINT:
        print(f"  warm-starting from: {WARM_START_CHECKPOINT}")
        model.load_state_dict(torch.load(WARM_START_CHECKPOINT, map_location=DEVICE))
    model.to(DEVICE)

    simulator = ForwardEulerSimulatorPop(model, cgm_min_scaled, cgm_max_scaled, ts=5.0).to(DEVICE)
    loss_fn = PopulationLoss(STATE_WEIGHTS, state_min, C1_INDEX, alpha=ALPHA, beta=BETA).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: LR_DECAY_PER_EPOCH ** epoch
    )

    train_sampler = FramedWindowSampler(x_train, u_train, SEQ_LEN, OVERLAP, BATCH_SIZE, DEVICE, seed=SEED)
    val_sampler = FramedWindowSampler(x_val, u_val, SEQ_LEN, OVERLAP, BATCH_SIZE, DEVICE, seed=SEED + 1)
    iters_per_epoch = train_sampler.n_batches_per_epoch()
    val_batches_per_epoch = val_sampler.n_batches_per_epoch()
    print(
        f"Windows available: train {len(train_sampler.pairs)} ({iters_per_epoch} batches/epoch if "
        f"visiting every window), val {len(val_sampler.pairs)} ({val_batches_per_epoch} batches/epoch) | "
        f"hop={train_sampler.hop} (75% overlap @ seq_len={SEQ_LEN})"
    )
    if MAX_TRAIN_BATCHES_PER_EPOCH is not None:
        iters_per_epoch = min(iters_per_epoch, MAX_TRAIN_BATCHES_PER_EPOCH)
    if MAX_VAL_BATCHES_PER_EPOCH is not None:
        val_batches_per_epoch = min(val_batches_per_epoch, MAX_VAL_BATCHES_PER_EPOCH)
    print(f"Using {iters_per_epoch} train batches/epoch and {val_batches_per_epoch} val batches/epoch "
          f"(capped by MAX_TRAIN_BATCHES_PER_EPOCH / MAX_VAL_BATCHES_PER_EPOCH).")

    # Select the validation subset ONCE, up front, and reuse the exact same batches
    # every single epoch (see get_fixed_subset's docstring for why this matters:
    # next_batch()'s persistent position would otherwise hand each epoch a different,
    # non-comparable slice of validation windows whenever val_batches_per_epoch is
    # capped below the full window count, corrupting early stopping / best-model
    # selection since val_loss would partly reflect which random slice was sampled).
    val_fixed_batches = val_sampler.get_fixed_subset(val_batches_per_epoch)
    print(f"Fixed validation subset: {len(val_fixed_batches)} batches, reused identically every epoch.")

    # --- quick timing probe: run a handful of real training batches up front so you get
    # an accurate seconds/batch and estimated-epoch-time number BEFORE committing to the
    # full loop below, instead of guessing from "it looks slow" ---
    WARMUP_BATCHES = min(20, iters_per_epoch)
    print(f"Timing probe: running {WARMUP_BATCHES} batches to estimate epoch duration ...")
    model.train()
    t_probe = time.time()
    for _ in range(WARMUP_BATCHES):
        optimizer.zero_grad()
        x0, u_batch, x_true = train_sampler.next_batch()
        x_sim = simulator(x0, u_batch)
        loss, l_fit, l_cons = loss_fn(x_sim, x_true, lim_inferior_scaled, lim_superior_scaled)
        loss.backward()
        optimizer.step()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()  # don't stop the clock until the GPU has actually finished
    probe_elapsed = time.time() - t_probe
    sec_per_batch = probe_elapsed / WARMUP_BATCHES
    est_epoch_sec = sec_per_batch * iters_per_epoch
    print(
        f"  {sec_per_batch:.3f} s/batch -> estimated ~{est_epoch_sec/60:.1f} min/epoch "
        f"for {iters_per_epoch} train batches (val batches will add roughly "
        f"{sec_per_batch * val_batches_per_epoch / 60:.1f} more min, forward-only so likely faster per batch)."
    )
    train_sampler._reshuffle()  # restart from a fresh shuffle so the warmup batches aren't skipped later

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_losses = []
        t_epoch = time.time()

        for it in range(iters_per_epoch):
            optimizer.zero_grad()
            x0, u_batch, x_true = train_sampler.next_batch()
            x_sim = simulator(x0, u_batch)

            if torch.isnan(x_sim).any() or torch.isinf(x_sim).any():
                print(f"WARNING: nan/inf in simulation at epoch {epoch} iter {it}, skipping batch")
                continue

            loss, l_fit, l_cons = loss_fn(x_sim, x_true, lim_inferior_scaled, lim_superior_scaled)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            # progress heartbeat: with epochs taking tens of minutes, printing only at
            # epoch-end leaves the terminal looking frozen for a long time. Print every
            # ~10% of the train loop (at least every 50 batches) so it's clear the run
            # is alive and roughly how far through the epoch it is.
            print_every = max(1, min(50, iters_per_epoch // 10))
            if (it + 1) % print_every == 0 or (it + 1) == iters_per_epoch:
                elapsed = time.time() - t_epoch
                running_mean = np.mean(epoch_losses) if epoch_losses else float("nan")
                print(
                    f"  epoch {epoch} [train {it+1}/{iters_per_epoch}] "
                    f"running_loss {running_mean:.6f} | {elapsed:.1f}s elapsed"
                )

        # --- validation pass over the FIXED subset selected once before training,
        # so val_loss is comparable across epochs (see val_fixed_batches comment above) ---
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_losses = []
            sq_err_sum = 0.0   # accumulates (y_sim - y_true)^2 in real mg/dL units, across ALL val batches
            n_err_vals = 0     # total count of individual timepoint errors summed into sq_err_sum
            t_val = time.time()
            print_every_val = max(1, min(50, val_batches_per_epoch // 10))
            for v_it, pair_batch in enumerate(val_fixed_batches):
                x0, u_batch, x_true = val_sampler.batch_from_pairs(pair_batch)
                x_sim = simulator(x0, u_batch)
                loss, l_fit, l_cons = loss_fn(x_sim, x_true, lim_inferior_scaled, lim_superior_scaled)
                val_losses.append(loss.item())

                # accumulate this batch's contribution to the mg/dL RMSE, rather than only
                # keeping the last batch's x_sim/x_true around after the loop ends
                y_sim_mgdl = scale_inverse_Q1(x_sim[1:, :, [0]].cpu().numpy(), OUTPUT_DIR)
                y_true_mgdl = scale_inverse_Q1(x_true[1:, :, [0]].cpu().numpy(), OUTPUT_DIR)
                batch_sq_err = (y_sim_mgdl - y_true_mgdl) ** 2
                sq_err_sum += batch_sq_err.sum()
                n_err_vals += batch_sq_err.size

                if (v_it + 1) % print_every_val == 0 or (v_it + 1) == val_batches_per_epoch:
                    print(f"  epoch {epoch} [val {v_it+1}/{val_batches_per_epoch}] "
                          f"{time.time()-t_val:.1f}s elapsed")

        mean_train_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        mean_val_loss = np.mean(val_losses) if val_losses else float("nan")

        # RMSE in real mg/dL units, averaged across every validation batch this epoch
        # (previously this only used the last batch's x_sim/x_true left over from the loop above)
        rmse_mgdl = np.sqrt(sq_err_sum / n_err_vals) if n_err_vals > 0 else float("nan")

        print(
            f"Epoch {epoch:4d} | train_loss {mean_train_loss:.6f} | "
            f"val_loss {mean_val_loss:.6f} | val_RMSE {rmse_mgdl:.2f} mg/dL | "
            f"lr {optimizer.param_groups[0]['lr']:.2e} | time {time.time()-t_epoch:.1f}s | "
            f"no_improve {epochs_without_improvement}/{PATIENCE}"
        )

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, MODEL_FILENAME))
            print(f"  -> new best val_loss, saved model to {os.path.join(OUTPUT_DIR, MODEL_FILENAME)}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(
                    f"Early stopping after {epochs_without_improvement} epochs "
                    f"without val_loss improvement."
                )
                break

    print("Training complete. Best model saved at:", os.path.join(OUTPUT_DIR, MODEL_FILENAME))


if __name__ == "__main__":
    main()