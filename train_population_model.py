"""
train_population_model.py  (fixed version)

Trains the population-level NN state-space model T1DSim_NN^P
(CGMOHSUSimStateSpaceModel_V2) on the merged simulated dataset produced by
merge_parts.py (population_development_dataset_merged.mat), following
Section 2.2.2 of Roquemen-Echeverri et al. (arXiv:2508.05705).

EXPECTED INPUT FILE (written by merge_parts.py):
    dataset_states  (N, T, 10)  order [Q1,Q2,S1,S2,I,X1,X2,X3,C2,C1]
    dataset_inputs  (N, T, 2)   [u_I (U/hr), u_carbs (g)]
    dataset_glucose (N, T)      mg/dL
    N is a multiple of 5; every 5 consecutive rows = one meal pattern.

WHAT CHANGED vs. the previous version
-------------------------------------
1. RAM: the merged file can hold ~231k traces (~18 GB of states). Added
   MAX_TRAIN_SCENARIOS / MAX_VAL_SCENARIOS, which cap how many traces are
   loaded. The cap is applied on whole 5-trace GROUPS, so no meal pattern is
   split. Without a cap the train buffer alone would need >11 GB.
2. Disk reads: for HDF5 (v7.3) files only the selected scenarios are read
   (contiguous runs), instead of streaming the entire file once per split.
3. RobustScaler is fit on a random subsample of rows (SCALER_FIT_ROWS)
   instead of every row (fitting on ~100M+ rows is very slow and needs many
   GB of temporary memory).
4. Window index table stored as int32 (half the memory).
5. Pickle files are closed properly.
6. Sanity check that channel 0 of the states really is Q1 in mmol/kg
   (Q1*18/0.16 should match dataset_glucose); warns if the file was not
   produced by merge_parts.py (wrong state order would silently corrupt S1).
7. INPUT_COLUMN_PERM knob in case your population model expects the input
   columns in a different order than [u_I, u_carbs].
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
OUTPUT_DIR = "models/PopulationModel/"
MODEL_FILENAME = "population_model_trained.pt"
WARM_START_CHECKPOINT = None

SEQ_LEN = 61
OVERLAP = 0.75
BATCH_SIZE = 128
MAX_EPOCHS = 150
PATIENCE = 20            # must be < MAX_EPOCHS to ever trigger

MAX_TRAIN_BATCHES_PER_EPOCH = 3000   # None = every window every epoch
MAX_VAL_BATCHES_PER_EPOCH = 1000     # None = every window every epoch
LR = 1e-3
LR_DECAY_PER_EPOCH = np.exp(-0.1)
WEIGHT_DECAY = 0.0
ALPHA = 0.7
BETA = 0.08
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2
SEED = 0

# --- memory knobs ---
# Max number of TRACES (rows) loaded per split. Rounded down to whole groups of 5.
# RAM per split ~= traces * T(2016) * 12 channels * 4 bytes ~= traces * 97 KB.
#   40000 traces ~= 3.9 GB | 8000 traces ~= 0.8 GB.  None = use the full split.
MAX_TRAIN_SCENARIOS = 40000
MAX_VAL_SCENARIOS = 8000
SCALER_FIT_ROWS = 4_000_000   # random rows used to fit the RobustScaler
CHUNK_SCENARIOS = 2000        # scenarios per disk read
CHUNK_ROWS = 500_000          # rows per scaler.transform() call

# If your population model expects input columns in a different order than
# [u_I, u_carbs], set e.g. [1, 0]. None = keep the file's order.
INPUT_COLUMN_PERM = None

STATE_ORDER = ["Q1", "Q2", "S1", "S2", "I", "X1", "X2", "X3", "C2", "C1"]
C1_INDEX = STATE_ORDER.index("C1")

STATE_WEIGHTS = np.array(
    [5/24, 1/6, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/24],
    dtype=np.float32,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator constants, used only for the Q1 sanity check
_VDG = 0.16
_MGDL_PER_MMOL = 18.0


# ==============================================================================
# Data loading
# ==============================================================================
def _load_var_scipy(path, key):
    d = sio.loadmat(path, variable_names=[key], simplify_cells=True)
    return d[key]


def _detect_backend(path):
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
    if backend == "scipy":
        shapes = {name: shape for name, shape, _ in sio.whosmat(path)}
        n_scenarios, T, n_states = shapes["dataset_states"]
        n_inputs = shapes["dataset_inputs"][-1]
        return n_scenarios, T, n_states, n_inputs
    with h5py.File(path, "r") as f:
        n_states, T, n_scenarios = f["dataset_states"].shape   # on disk (10,T,N)
        n_inputs = f["dataset_inputs"].shape[0]                # on disk (2,T,N)
    return n_scenarios, T, n_states, n_inputs


_q1_checked = False


def _check_q1_channel(states_chunk, glucose_chunk):
    """Warn once if channel 0 doesn't look like Q1 (mmol/kg) matching glucose."""
    global _q1_checked
    if _q1_checked:
        return
    _q1_checked = True
    implied = states_chunk[..., 0] * _MGDL_PER_MMOL / _VDG
    err = float(np.abs(implied - glucose_chunk).max())
    if err > 1.0:
        print(f"  WARNING: channel 0 of dataset_states does not match Q1*18/0.16 vs "
              f"dataset_glucose (max abs diff {err:.1f} mg/dL). If this file was not made by "
              f"merge_parts.py, the state order may be wrong (channel 0 must be Q1).")
    else:
        print(f"  Q1 sanity check OK (max diff {err:.3f} mg/dL) - state order looks correct.")


def _iter_raw_chunks_scipy(path, chunk_size):
    dataset_states = _load_var_scipy(path, "dataset_states")
    dataset_inputs = _load_var_scipy(path, "dataset_inputs")
    dataset_glucose = _load_var_scipy(path, "dataset_glucose")
    N = dataset_states.shape[0]
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        states_chunk = dataset_states[start:end].astype(np.float32)
        inputs_chunk = dataset_inputs[start:end].astype(np.float32)
        glucose_chunk = dataset_glucose[start:end].astype(np.float32)
        _check_q1_channel(states_chunk, glucose_chunk)
        states_chunk[..., 0] = glucose_chunk
        yield start, end, states_chunk, inputs_chunk


def fill_split_buffer(path, backend, idx, T, n_states, n_inputs, chunk_size=CHUNK_SCENARIOS):
    """Loads only scenarios `idx` (sorted ascending) into flat float32 buffers
    of shape (len(idx)*T, n_states) / (len(idx)*T, n_inputs)."""
    idx = np.asarray(idx)
    assert np.all(np.diff(idx) > 0), "idx must be strictly ascending"

    x_flat = np.empty((len(idx) * T, n_states), dtype=np.float32)
    u_flat = np.empty((len(idx) * T, n_inputs), dtype=np.float32)

    if backend == "h5py":
        # Read only the needed scenarios, as contiguous runs (each capped at chunk_size).
        runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        pos = 0
        with h5py.File(path, "r") as f:
            ds_s, ds_i, ds_g = f["dataset_states"], f["dataset_inputs"], f["dataset_glucose"]
            for run in runs:
                run_start, run_end = int(run[0]), int(run[-1]) + 1
                for a in range(run_start, run_end, chunk_size):
                    b = min(a + chunk_size, run_end)
                    c = b - a
                    st = np.transpose(ds_s[:, :, a:b], (2, 1, 0)).astype(np.float32, order="C")
                    ip = np.transpose(ds_i[:, :, a:b], (2, 1, 0)).astype(np.float32, order="C")
                    gl = np.transpose(ds_g[:, a:b], (1, 0)).astype(np.float32, order="C")
                    _check_q1_channel(st, gl)
                    st[..., 0] = gl
                    x_flat[pos * T:(pos + c) * T] = st.reshape(c * T, n_states)
                    u_flat[pos * T:(pos + c) * T] = ip.reshape(c * T, n_inputs)
                    pos += c
        assert pos == len(idx)
        return x_flat, u_flat

    # scipy backend (legacy non-v7.3 files): must load whole variables once
    pos_of = {raw_i: split_pos for split_pos, raw_i in enumerate(idx)}
    n_found = 0
    for start, end, states_chunk, inputs_chunk in _iter_raw_chunks_scipy(path, chunk_size):
        in_range = idx[(idx >= start) & (idx < end)]
        for raw_i in in_range:
            local = raw_i - start
            r0 = pos_of[raw_i] * T
            x_flat[r0:r0 + T] = states_chunk[local]
            u_flat[r0:r0 + T] = inputs_chunk[local]
            n_found += 1
        if n_found == len(idx):
            break
    return x_flat, u_flat


def transform_in_place(scaler, flat_array, chunk_rows=CHUNK_ROWS):
    n = flat_array.shape[0]
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        flat_array[start:end] = scaler.transform(flat_array[start:end]).astype(np.float32)


def group_split(n_scenarios, group_size=5, train_frac=0.6, val_frac=0.2, seed=0,
                max_train=None, max_val=None):
    """Train/val/test split over groups of `group_size` consecutive rows (one
    meal pattern). max_train / max_val cap the number of ROWS per split
    (rounded down to whole groups)."""
    assert n_scenarios % group_size == 0, (
        f"n_scenarios ({n_scenarios}) isn't divisible by group_size ({group_size}) - "
        "re-run merge_parts.py (use --rare drop, or keep with padding)."
    )
    n_groups = n_scenarios // group_size
    rng = np.random.RandomState(seed)
    group_order = rng.permutation(n_groups)

    n_train_groups = int(round(n_groups * train_frac))
    n_val_groups = int(round(n_groups * val_frac))

    train_groups = group_order[:n_train_groups]
    val_groups = group_order[n_train_groups:n_train_groups + n_val_groups]
    test_groups = group_order[n_train_groups + n_val_groups:]

    if max_train is not None:
        train_groups = train_groups[: max(1, max_train // group_size)]
    if max_val is not None:
        val_groups = val_groups[: max(1, max_val // group_size)]

    def groups_to_indices(groups):
        idx = (groups[:, None] * group_size + np.arange(group_size)[None, :]).reshape(-1)
        return np.sort(idx)

    return (groups_to_indices(train_groups),
            groups_to_indices(val_groups),
            groups_to_indices(test_groups))


# ==============================================================================
# Framed batch sampler (true 75% overlap, index pairs only)
# ==============================================================================
class FramedWindowSampler:
    def __init__(self, x_est, u_fit, seq_len, overlap, batch_size, device, seed=0):
        self.x_est = x_est
        self.u_fit = u_fit
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.N, self.T, self.n_states = x_est.shape

        self.hop = max(1, int((1 - overlap) * seq_len))
        starts = np.arange(0, self.T - self.seq_len + 1, self.hop)

        scenario_grid, start_grid = np.meshgrid(np.arange(self.N), starts, indexing="ij")
        self.pairs = np.stack([scenario_grid.ravel(), start_grid.ravel()], axis=1).astype(np.int32)
        del scenario_grid, start_grid

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
        chosen = self.epoch_order[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        return self.batch_from_pairs(self.pairs[chosen])

    def get_fixed_subset(self, n_batches):
        """Fixed set of validation batches, chosen once and replayed each epoch so
        val_loss is comparable across epochs."""
        n_needed = min(n_batches * self.batch_size, len(self.pairs))
        chosen = self.rng.permutation(len(self.pairs))[:n_needed]
        return [self.pairs[chosen[s:s + self.batch_size]]
                for s in range(0, len(chosen), self.batch_size)]

    def batch_from_pairs(self, pair_batch):
        scenario_idx = pair_batch[:, 0]
        start_idx = pair_batch[:, 1]
        idx_range = start_idx[:, None] + np.arange(self.seq_len)[None, :]

        x_batch = self.x_est[scenario_idx[:, None], idx_range]   # (B, seq_len, 10)
        u_batch = self.u_fit[scenario_idx[:, None], idx_range]   # (B, seq_len, 2)

        x_batch = np.transpose(x_batch, (1, 0, 2))
        u_batch = np.transpose(u_batch, (1, 0, 2))
        x0 = x_batch[0]

        return (
            torch.tensor(x0, dtype=torch.float32, device=self.device),
            torch.tensor(u_batch, dtype=torch.float32, device=self.device),
            torch.tensor(x_batch, dtype=torch.float32, device=self.device),
        )


# ==============================================================================
# Simulator
# ==============================================================================
class ForwardEulerSimulatorPop(nn.Module):
    def __init__(self, ss_pop_model, cgm_min, cgm_max, ts=5.0):
        super().__init__()
        self.ss_pop_model = ss_pop_model
        self.ts = ts
        self.cgm_min = cgm_min
        self.cgm_max = cgm_max

    def adjust_cgm(self, x):
        return torch.clamp(x, min=self.cgm_min, max=self.cgm_max)

    def forward(self, x0_batch, u_batch):
        X_sim_list = []
        x_step = x0_batch
        for step in range(u_batch.shape[0]):
            u_step = u_batch[step]
            X_sim_list.append(x_step)
            dx = self.ss_pop_model(x_step, u_step)
            x_step = x_step + self.ts * dx
            x_step = torch.cat([self.adjust_cgm(x_step[:, [0]]), x_step[:, 1:]], dim=1)
        return torch.stack(X_sim_list, 0)


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
        penalty = torch.ones_like(y_true)
        penalty[torch.logical_and(y_true <= lim_inferior, y_sim > y_true)] = 2.0
        penalty[torch.logical_and(y_true >= lim_superior, y_sim < y_true)] = 6.0
        return penalty

    def __call__(self, x_sim, x_true, lim_inferior_scaled, lim_superior_scaled):
        y_sim = x_sim[1:, :, [0]]
        y_true = x_true[1:, :, [0]]
        penalty = self.fit_penalty(y_sim, y_true, lim_inferior_scaled, lim_superior_scaled)
        L_fit = torch.mean((y_sim - y_true) ** 2 * penalty)

        err = x_sim[1:] - x_true[1:]
        mse_per_state = torch.mean(err ** 2, dim=(0, 1))

        floor = self.state_min.clone()
        floor[self.c1_index] = 0.0
        soft_constraint = torch.clamp(-(x_sim[1:] - floor), min=0.0)
        soft_penalty_per_state = torch.sum(soft_constraint, dim=(0, 1))

        per_state_loss = mse_per_state + self.beta * soft_penalty_per_state
        L_consistency = torch.sum(self.w * per_state_loss)

        L_total = L_fit + self.alpha * L_consistency
        return L_total, L_fit.item(), L_consistency.item()


# ==============================================================================
# Main
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
          f"({n_scenarios}, {T}, {n_inputs}) inputs [{time.time()-t0:.2f}s]")

    print("Splitting scenarios into train/val/test (grouped by meal pattern) ...")
    train_idx, val_idx, test_idx = group_split(
        n_scenarios, group_size=5, train_frac=TRAIN_FRACTION, val_frac=VAL_FRACTION, seed=SEED,
        max_train=MAX_TRAIN_SCENARIOS, max_val=MAX_VAL_SCENARIOS,
    )
    est_gb = (len(train_idx) + len(val_idx)) * T * (n_states + n_inputs) * 4 / 1e9
    print(f"  train: {len(train_idx)} | val: {len(val_idx)} | test (unused, never read): {len(test_idx)} "
          f"| approx RAM for data: {est_gb:.1f} GB")

    print(f"Loading TRAIN scenarios (chunks of {CHUNK_SCENARIOS}) ...")
    t0 = time.time()
    x_train_flat, u_train_flat = fill_split_buffer(
        MERGED_MAT_PATH, backend, train_idx, T, n_states, n_inputs)
    if INPUT_COLUMN_PERM is not None:
        u_train_flat = np.ascontiguousarray(u_train_flat[:, INPUT_COLUMN_PERM])
    print(f"  done in {time.time()-t0:.1f}s | x {x_train_flat.nbytes/1e9:.2f} GB, "
          f"u {u_train_flat.nbytes/1e9:.2f} GB")

    print(f"Fitting RobustScaler on a {SCALER_FIT_ROWS:,}-row random subsample of TRAIN ...")
    from sklearn.preprocessing import RobustScaler
    n_rows = x_train_flat.shape[0]
    rng_fit = np.random.RandomState(SEED)
    if n_rows > SCALER_FIT_ROWS:
        fit_rows = np.sort(rng_fit.choice(n_rows, SCALER_FIT_ROWS, replace=False))
    else:
        fit_rows = np.arange(n_rows)
    scaler_states = RobustScaler().fit(x_train_flat[fit_rows])
    scaler_inputs = RobustScaler().fit(u_train_flat[fit_rows])
    del fit_rows

    with open(os.path.join(OUTPUT_DIR, "scaler_states.pkl"), "wb") as fh:
        dump(scaler_states, fh)
    with open(os.path.join(OUTPUT_DIR, "scaler_inputs.pkl"), "wb") as fh:
        dump(scaler_inputs, fh)
    print(f"  saved scalers to {OUTPUT_DIR}")

    print("Scaling TRAIN in place ...")
    transform_in_place(scaler_states, x_train_flat)
    transform_in_place(scaler_inputs, u_train_flat)
    x_train = x_train_flat.reshape(len(train_idx), T, n_states)
    u_train = u_train_flat.reshape(len(train_idx), T, n_inputs)

    print(f"Loading VAL scenarios ...")
    t0 = time.time()
    x_val_flat, u_val_flat = fill_split_buffer(
        MERGED_MAT_PATH, backend, val_idx, T, n_states, n_inputs)
    if INPUT_COLUMN_PERM is not None:
        u_val_flat = np.ascontiguousarray(u_val_flat[:, INPUT_COLUMN_PERM])
    print(f"  done in {time.time()-t0:.1f}s")

    print("Scaling VAL in place ...")
    transform_in_place(scaler_states, x_val_flat)
    transform_in_place(scaler_inputs, u_val_flat)
    x_val = x_val_flat.reshape(len(val_idx), T, n_states)
    u_val = u_val_flat.reshape(len(val_idx), T, n_inputs)

    state_min = x_train.reshape(-1, n_states).min(axis=0)

    lim_inferior_scaled = scale_single_state(70, "Q1", OUTPUT_DIR)
    lim_superior_scaled = scale_single_state(250, "Q1", OUTPUT_DIR)
    cgm_min_scaled = scale_single_state(40, "Q1", OUTPUT_DIR)
    cgm_max_scaled = scale_single_state(400, "Q1", OUTPUT_DIR)

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
        optimizer, lr_lambda=lambda epoch: LR_DECAY_PER_EPOCH ** epoch)

    train_sampler = FramedWindowSampler(x_train, u_train, SEQ_LEN, OVERLAP, BATCH_SIZE, DEVICE, seed=SEED)
    val_sampler = FramedWindowSampler(x_val, u_val, SEQ_LEN, OVERLAP, BATCH_SIZE, DEVICE, seed=SEED + 1)
    iters_per_epoch = train_sampler.n_batches_per_epoch()
    val_batches_per_epoch = val_sampler.n_batches_per_epoch()
    print(f"Windows: train {len(train_sampler.pairs)} ({iters_per_epoch} batches), "
          f"val {len(val_sampler.pairs)} ({val_batches_per_epoch} batches) | hop={train_sampler.hop}")
    if MAX_TRAIN_BATCHES_PER_EPOCH is not None:
        iters_per_epoch = min(iters_per_epoch, MAX_TRAIN_BATCHES_PER_EPOCH)
    if MAX_VAL_BATCHES_PER_EPOCH is not None:
        val_batches_per_epoch = min(val_batches_per_epoch, MAX_VAL_BATCHES_PER_EPOCH)
    print(f"Using {iters_per_epoch} train / {val_batches_per_epoch} val batches per epoch.")

    val_fixed_batches = val_sampler.get_fixed_subset(val_batches_per_epoch)
    print(f"Fixed validation subset: {len(val_fixed_batches)} batches, reused every epoch.")

    # Timing probe (does real optimizer steps on a few batches; fine as warm-up)
    WARMUP_BATCHES = min(20, iters_per_epoch)
    print(f"Timing probe: {WARMUP_BATCHES} batches ...")
    model.train()
    t_probe = time.time()
    for _ in range(WARMUP_BATCHES):
        optimizer.zero_grad()
        x0, u_batch, x_true = train_sampler.next_batch()
        x_sim = simulator(x0, u_batch)
        loss, _, _ = loss_fn(x_sim, x_true, lim_inferior_scaled, lim_superior_scaled)
        loss.backward()
        optimizer.step()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    sec_per_batch = (time.time() - t_probe) / WARMUP_BATCHES
    print(f"  {sec_per_batch:.3f} s/batch -> ~{sec_per_batch * iters_per_epoch / 60:.1f} min/epoch (train) "
          f"+ ~{sec_per_batch * val_batches_per_epoch / 60:.1f} min val")
    train_sampler._reshuffle()

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_losses = []
        t_epoch = time.time()
        print_every = max(1, min(50, iters_per_epoch // 10))

        for it in range(iters_per_epoch):
            optimizer.zero_grad()
            x0, u_batch, x_true = train_sampler.next_batch()
            x_sim = simulator(x0, u_batch)

            if torch.isnan(x_sim).any() or torch.isinf(x_sim).any():
                print(f"WARNING: nan/inf in simulation at epoch {epoch} iter {it}, skipping batch")
                continue

            loss, _, _ = loss_fn(x_sim, x_true, lim_inferior_scaled, lim_superior_scaled)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            if (it + 1) % print_every == 0 or (it + 1) == iters_per_epoch:
                running_mean = np.mean(epoch_losses) if epoch_losses else float("nan")
                print(f"  epoch {epoch} [train {it+1}/{iters_per_epoch}] "
                      f"running_loss {running_mean:.6f} | {time.time()-t_epoch:.1f}s elapsed")

        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_losses = []
            sq_err_sum = 0.0
            n_err_vals = 0
            t_val = time.time()
            print_every_val = max(1, min(50, len(val_fixed_batches) // 10))
            for v_it, pair_batch in enumerate(val_fixed_batches):
                x0, u_batch, x_true = val_sampler.batch_from_pairs(pair_batch)
                x_sim = simulator(x0, u_batch)
                loss, _, _ = loss_fn(x_sim, x_true, lim_inferior_scaled, lim_superior_scaled)
                val_losses.append(loss.item())

                y_sim_mgdl = scale_inverse_Q1(x_sim[1:, :, [0]].cpu().numpy(), OUTPUT_DIR)
                y_true_mgdl = scale_inverse_Q1(x_true[1:, :, [0]].cpu().numpy(), OUTPUT_DIR)
                batch_sq_err = (y_sim_mgdl - y_true_mgdl) ** 2
                sq_err_sum += batch_sq_err.sum()
                n_err_vals += batch_sq_err.size

                if (v_it + 1) % print_every_val == 0 or (v_it + 1) == len(val_fixed_batches):
                    print(f"  epoch {epoch} [val {v_it+1}/{len(val_fixed_batches)}] "
                          f"{time.time()-t_val:.1f}s elapsed")

        mean_train_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        mean_val_loss = np.mean(val_losses) if val_losses else float("nan")
        rmse_mgdl = np.sqrt(sq_err_sum / n_err_vals) if n_err_vals > 0 else float("nan")

        print(f"Epoch {epoch:4d} | train_loss {mean_train_loss:.6f} | val_loss {mean_val_loss:.6f} | "
              f"val_RMSE {rmse_mgdl:.2f} mg/dL | lr {optimizer.param_groups[0]['lr']:.2e} | "
              f"time {time.time()-t_epoch:.1f}s | no_improve {epochs_without_improvement}/{PATIENCE}")

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, MODEL_FILENAME))
            print(f"  -> new best val_loss, saved to {os.path.join(OUTPUT_DIR, MODEL_FILENAME)}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping after {epochs_without_improvement} epochs without improvement.")
                break

    print("Training complete. Best model saved at:", os.path.join(OUTPUT_DIR, MODEL_FILENAME))


if __name__ == "__main__":
    main()
