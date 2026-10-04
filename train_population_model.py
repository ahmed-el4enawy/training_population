"""
train_population_model.py  (fixed version)

Trains the population-level NN state-space model T1DSim_NN^P
(CGMOHSUSimStateSpaceModel_V2) on the merged simulated dataset produced by
merge_parts.py (population_development_dataset_merged.mat), following
Section 2.2.2 of Roquemen-Echeverri et al. (arXiv:2508.05705).

EXPECTED INPUT FILE (written by merge_parts.py):
    dataset_states   (N, T, 10)  order [Q1,Q2,S1,S2,I,X1,X2,X3,C2,C1]
    dataset_inputs   (N, T, 2)   [u_I (U/hr), u_carbs (g)]
    dataset_glucose  (N, T)      mg/dL
    dataset_split_id (N,)        0=Train, 1=Val, 2=Test
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pickle import dump
import h5py
from sklearn.preprocessing import RobustScaler

from t1dsim_ai.population_model import CGMOHSUSimStateSpaceModel_V2
from t1dsim_ai.options import n_neurons_pop
from t1dsim_ai.utils.preprocess import scale_single_state, scale_inverse_Q1

# ==============================================================================
# CONFIG - EDIT THESE
# ==============================================================================
MERGED_MAT_PATH = r"C:\Users\pc\Downloads\population\population_development_dataset_merged.mat"
OUTPUT_DIR = "models/PopulationModel_v2/"     
MODEL_FILENAME = "population_model_trained.pt"
WARM_START_CHECKPOINT = None

# [PAPER-EXPLICIT]
SEQ_LEN = 61
TRAIN_OVERLAP = 0.75
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 0.0 # [OFFICIAL-CODE-DERIVED]
ALPHA = 0.7
BETA = 0.08
# Paper says e^-0.1 per epoch, which we apply after every FULL pass.
LR_DECAY_PER_EPOCH = np.exp(-0.1)

# [UNRESOLVED]
# The paper explicitly provides 150 epochs for individual models, but omits the population count.
# Official artifact analysis shows "epoch_15" checkpoints.
# Set these manually before running training.
MAX_EPOCHS = None
PATIENCE = None
VAL_OVERLAP = None
CHECKPOINT_POLICY = None # "final_epoch" or "best_validation"

# [ENGINEERING] (Memory management limits)
# If RAM is insufficient to load all data or fit scalers, these cap the usage.
# To run a strict reproduction using all data, set these to None.
MAX_TRAIN_SCENARIOS = None
MAX_VAL_SCENARIOS = None
MAX_TRAIN_BATCHES_PER_EPOCH = None
MAX_VAL_BATCHES_PER_EPOCH = None
SCALER_FIT_ROWS = None
CHUNK_SCENARIOS = 2000

INPUT_COLUMN_PERM = None
STATE_ORDER = ["Q1", "Q2", "S1", "S2", "I", "X1", "X2", "X3", "C2", "C1"]
C1_INDEX = STATE_ORDER.index("C1")

# [RECONSTRUCTION] Exact rational fractions recreating paper's rounded decimals
STATE_WEIGHTS = np.array(
    [5/24, 1/6, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/24],
    dtype=np.float32,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_VDG = 0.16
_MGDL_PER_MMOL = 18.0

# ==============================================================================
# Data loading
# ==============================================================================
def get_dataset_dims(path):
    with h5py.File(path, "r") as f:
        n_states, T, n_scenarios = f["dataset_states"].shape   # on disk (10,T,N)
        n_inputs = f["dataset_inputs"].shape[0]                # on disk (2,T,N)
    return n_scenarios, T, n_states, n_inputs

def get_splits(path, max_train=None, max_val=None):
    with h5py.File(path, "r") as f:
        split_id = f["dataset_split_id"][()]
    train_idx = np.where(split_id == 0)[0]
    val_idx = np.where(split_id == 1)[0]
    test_idx = np.where(split_id == 2)[0]
    
    if max_train is not None: train_idx = train_idx[:max_train]
    if max_val is not None: val_idx = val_idx[:max_val]
    return train_idx, val_idx, test_idx

_q1_checked = False
def _check_q1_channel(states_chunk, glucose_chunk):
    global _q1_checked
    if _q1_checked: return
    _q1_checked = True
    implied = states_chunk[..., 0] * _MGDL_PER_MMOL / _VDG
    err = float(np.abs(implied - glucose_chunk).max())
    if err > 1.0:
        raise RuntimeError(f"Strict mode failure: channel 0 does not match Q1 (max diff {err:.1f} mg/dL).")

def fill_split_buffer(path, idx, T, n_states, n_inputs, chunk_size=CHUNK_SCENARIOS):
    idx = np.asarray(idx)
    x_flat = np.empty((len(idx) * T, n_states), dtype=np.float32)
    u_flat = np.empty((len(idx) * T, n_inputs), dtype=np.float32)

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
    return x_flat, u_flat

def transform_in_place(scaler, flat_array, chunk_rows=500_000):
    n = flat_array.shape[0]
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        flat_array[start:end] = scaler.transform(flat_array[start:end]).astype(np.float32)

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

        self.hop = max(1, int(round((1 - overlap) * (seq_len - 1))))
        starts = np.arange(0, self.T - self.seq_len + 1, self.hop)

        scenario_grid, start_grid = np.meshgrid(np.arange(self.N), starts, indexing="ij")
        self.pairs = np.stack([scenario_grid.ravel(), start_grid.ravel()], axis=1).astype(np.int32)
        
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
        n_needed = min(n_batches * self.batch_size, len(self.pairs))
        chosen = self.rng.permutation(len(self.pairs))[:n_needed]
        return [self.pairs[chosen[s:s + self.batch_size]]
                for s in range(0, len(chosen), self.batch_size)]

    def batch_from_pairs(self, pair_batch):
        scenario_idx = pair_batch[:, 0]
        start_idx = pair_batch[:, 1]
        idx_range = start_idx[:, None] + np.arange(self.seq_len)[None, :]

        x_batch = self.x_est[scenario_idx[:, None], idx_range]
        u_batch = self.u_fit[scenario_idx[:, None], idx_range]

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
    def __init__(self, ss_pop_model, ts=5.0, cgm_min=None, cgm_max=None):
        super().__init__()
        self.ss_pop_model = ss_pop_model
        self.ts = ts
        self.cgm_min = cgm_min
        self.cgm_max = cgm_max

    def adjust_cgm(self, x):
        # [UNRESOLVED] The paper does not specify clamping Q1. 
        # But if stability requires it, we can clamp it. Otherwise return x.
        if self.cgm_min is not None and self.cgm_max is not None:
            return torch.clamp(x, min=self.cgm_min, max=self.cgm_max)
        return x

    def forward(self, x0_batch, u_batch):
        X_sim_list = [x0_batch]
        x_step = x0_batch
        # u_batch shape: (seq_len, B, 2)
        # 60 integration intervals for 61 samples.
        for step in range(u_batch.shape[0] - 1):
            u_step = u_batch[step]
            dx = self.ss_pop_model(x_step, u_step)
            x_step = x_step + self.ts * dx
            x_step = torch.cat([self.adjust_cgm(x_step[:, [0]]), x_step[:, 1:]], dim=1)
            X_sim_list.append(x_step)
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
        # Initial state is x_sim[0], which equals x_true[0]. 
        # Compare prediction targets x_sim[1:] to x_true[1:]
        y_sim = x_sim[1:, :, [0]]
        y_true = x_true[1:, :, [0]]
        penalty = self.fit_penalty(y_sim, y_true, lim_inferior_scaled, lim_superior_scaled)
        L_fit = torch.mean((y_sim - y_true) ** 2 * penalty)

        err = x_sim[1:] - x_true[1:]
        mse_per_state = torch.mean(err ** 2, dim=(0, 1))

        # Enforce MSE^{C_1} = 0 as per Section 2.2.1
        mask = torch.ones_like(mse_per_state)
        mask[self.c1_index] = 0.0
        mse_per_state = mse_per_state * mask

        # Soft minimum floor penalty
        floor = self.state_min
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

    if MAX_EPOCHS is None:
        raise ValueError("MAX_EPOCHS is explicitly [UNRESOLVED] and must be configured manually before training.")
    if CHECKPOINT_POLICY not in ["best_validation", "final_epoch"]:
        raise ValueError("CHECKPOINT_POLICY must be configured as 'best_validation' or 'final_epoch'.")
    if VAL_OVERLAP is None:
        raise ValueError("VAL_OVERLAP must be configured manually before training.")
    print(f"Inspecting merged dataset at: {MERGED_MAT_PATH}")
    t0 = time.time()
    n_scenarios, T, n_states, n_inputs = get_dataset_dims(MERGED_MAT_PATH)
    print(f"  shape: ({n_scenarios}, {T}, {n_states}) states, "
          f"({n_scenarios}, {T}, {n_inputs}) inputs [{time.time()-t0:.2f}s]")

    print("Extracting pre-existing Train/Val/Test splits ...")
    train_idx, val_idx, test_idx = get_splits(
        MERGED_MAT_PATH, max_train=MAX_TRAIN_SCENARIOS, max_val=MAX_VAL_SCENARIOS
    )
    est_gb = (len(train_idx) + len(val_idx)) * T * (n_states + n_inputs) * 4 / 1e9
    print(f"  train: {len(train_idx)} | val: {len(val_idx)} | test (unused): {len(test_idx)} "
          f"| approx RAM: {est_gb:.1f} GB")

    print(f"Loading TRAIN scenarios ...")
    t0 = time.time()
    x_train_flat, u_train_flat = fill_split_buffer(MERGED_MAT_PATH, train_idx, T, n_states, n_inputs)
    if INPUT_COLUMN_PERM is not None:
        u_train_flat = np.ascontiguousarray(u_train_flat[:, INPUT_COLUMN_PERM])
    print(f"  done in {time.time()-t0:.1f}s")

    print("Fitting RobustScaler on TRAIN ...")
    n_rows = x_train_flat.shape[0]
    rng_fit = np.random.RandomState(SEED)
    if SCALER_FIT_ROWS and n_rows > SCALER_FIT_ROWS:
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
    x_val_flat, u_val_flat = fill_split_buffer(MERGED_MAT_PATH, val_idx, T, n_states, n_inputs)
    if INPUT_COLUMN_PERM is not None:
        u_val_flat = np.ascontiguousarray(u_val_flat[:, INPUT_COLUMN_PERM])

    print("Scaling VAL in place ...")
    transform_in_place(scaler_states, x_val_flat)
    transform_in_place(scaler_inputs, u_val_flat)
    x_val = x_val_flat.reshape(len(val_idx), T, n_states)
    u_val = u_val_flat.reshape(len(val_idx), T, n_inputs)

    state_min = x_train.reshape(-1, n_states).min(axis=0)
    state_min[C1_INDEX] = (0.0 - scaler_states.center_[C1_INDEX]) / scaler_states.scale_[C1_INDEX]

    lim_inferior_scaled = scale_single_state(70, "Q1", OUTPUT_DIR)
    lim_superior_scaled = scale_single_state(250, "Q1", OUTPUT_DIR)
    
    # [UNRESOLVED] Explicit Q1 bounds during population Euler rollout are not in paper.
    # Official codebase sets cgm_min=40, cgm_max=400 in T1DSimODE but not population_model.
    # Leaving None for strict adherence, or can be added if NaN issues arise.
    cgm_min_scaled = None # scale_single_state(40, "Q1", OUTPUT_DIR)
    cgm_max_scaled = None # scale_single_state(400, "Q1", OUTPUT_DIR)

    print(f"Building model on device: {DEVICE}")
    model = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
    if WARM_START_CHECKPOINT:
        model.load_state_dict(torch.load(WARM_START_CHECKPOINT, map_location=DEVICE))
    model.to(DEVICE)

    simulator = ForwardEulerSimulatorPop(model, ts=5.0, cgm_min=cgm_min_scaled, cgm_max=cgm_max_scaled).to(DEVICE)
    loss_fn = PopulationLoss(STATE_WEIGHTS, state_min, C1_INDEX, alpha=ALPHA, beta=BETA).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_sampler = FramedWindowSampler(x_train, u_train, SEQ_LEN, TRAIN_OVERLAP, BATCH_SIZE, DEVICE, seed=SEED)
    val_sampler = FramedWindowSampler(x_val, u_val, SEQ_LEN, VAL_OVERLAP, BATCH_SIZE, DEVICE, seed=SEED + 1)
    
    iters_per_epoch = train_sampler.n_batches_per_epoch()
    val_batches_per_epoch = val_sampler.n_batches_per_epoch()
    
    if MAX_TRAIN_BATCHES_PER_EPOCH is not None:
        iters_per_epoch = min(iters_per_epoch, MAX_TRAIN_BATCHES_PER_EPOCH)
    if MAX_VAL_BATCHES_PER_EPOCH is not None:
        val_batches_per_epoch = min(val_batches_per_epoch, MAX_VAL_BATCHES_PER_EPOCH)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY_PER_EPOCH)
    val_fixed_batches = val_sampler.get_fixed_subset(val_batches_per_epoch)

    # Benchmarking without mutating parameters
    WARMUP_BATCHES = min(20, iters_per_epoch)
    print(f"Timing probe: {WARMUP_BATCHES} batches (non-mutating) ...")
    model.eval()
    t_probe = time.time()
    probe_pos = train_sampler.pos
    probe_epoch_order = train_sampler.epoch_order.copy()
    probe_rng_state = train_sampler.rng.get_state()
    
    with torch.no_grad():
        for _ in range(WARMUP_BATCHES):
            x0, u_batch, x_true = train_sampler.next_batch()
            x_sim = simulator(x0, u_batch)
            loss_fn(x_sim, x_true, lim_inferior_scaled, lim_superior_scaled)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        
    train_sampler.rng.set_state(probe_rng_state)
    train_sampler.epoch_order = probe_epoch_order
    train_sampler.pos = probe_pos
    
    sec_per_batch = (time.time() - t_probe) / max(1, WARMUP_BATCHES)
    print(f"  {sec_per_batch:.3f} s/batch")

    train_sampler._reshuffle()
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    skipped_total = 0
    nonfinite_sim_batches = 0
    nonfinite_loss_batches = 0
    nonfinite_grad_batches = 0

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
                nonfinite_sim_batches += 1
                skipped_total += 1
                continue

            loss, _, _ = loss_fn(x_sim, x_true, lim_inferior_scaled, lim_superior_scaled)
            if not torch.isfinite(loss):
                nonfinite_loss_batches += 1
                skipped_total += 1
                continue
                
            loss.backward()
            
            # NaN gradient safety
            grad_is_nan = False
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    grad_is_nan = True
                    break
            
            if grad_is_nan:
                nonfinite_grad_batches += 1
                skipped_total += 1
                optimizer.zero_grad()
                continue
                
            optimizer.step()
            epoch_losses.append(loss.item())

            if (it + 1) % print_every == 0 or (it + 1) == iters_per_epoch:
                running_mean = np.mean(epoch_losses) if epoch_losses else float("nan")
                print(f"  epoch {epoch} [train {it+1}/{iters_per_epoch}] "
                      f"loss {running_mean:.6f} | {time.time()-t_epoch:.1f}s")

        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_losses = []
            sq_err_sum = 0.0
            n_err_vals = 0
            for v_it, pair_batch in enumerate(val_fixed_batches):
                x0, u_batch, x_true = val_sampler.batch_from_pairs(pair_batch)
                x_sim = simulator(x0, u_batch)
                if torch.isnan(x_sim).any() or torch.isinf(x_sim).any():
                    continue
                    
                loss, _, _ = loss_fn(x_sim, x_true, lim_inferior_scaled, lim_superior_scaled)
                val_losses.append(loss.item())

                y_sim_mgdl = scale_inverse_Q1(x_sim[1:, :, [0]].cpu().numpy(), OUTPUT_DIR)
                y_true_mgdl = scale_inverse_Q1(x_true[1:, :, [0]].cpu().numpy(), OUTPUT_DIR)
                batch_sq_err = (y_sim_mgdl - y_true_mgdl) ** 2
                sq_err_sum += batch_sq_err.sum()
                n_err_vals += batch_sq_err.size

        mean_train_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        mean_val_loss = np.mean(val_losses) if val_losses else float("nan")
        rmse_mgdl = np.sqrt(sq_err_sum / n_err_vals) if n_err_vals > 0 else float("nan")

        print(f"Epoch {epoch:4d} | train_loss {mean_train_loss:.6f} | val_loss {mean_val_loss:.6f} | "
              f"val_RMSE {rmse_mgdl:.2f} mg/dL | lr {optimizer.param_groups[0]['lr']:.2e} | "
              f"time {time.time()-t_epoch:.1f}s")

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            epochs_without_improvement = 0
            if CHECKPOINT_POLICY == "best_validation":
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, MODEL_FILENAME))
                print(f"  -> new best val_loss, saved to {os.path.join(OUTPUT_DIR, MODEL_FILENAME)}")
        else:
            epochs_without_improvement += 1
            if PATIENCE is not None and epochs_without_improvement >= PATIENCE:
                print(f"Early stopping after {epochs_without_improvement} epochs.")
                break
                
        if CHECKPOINT_POLICY == "final_epoch":
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, MODEL_FILENAME))
            print(f"  -> saved final_epoch checkpoint to {os.path.join(OUTPUT_DIR, MODEL_FILENAME)}")

    print(f"Training complete. Skipped {skipped_total} NaN/Inf batches (Sim: {nonfinite_sim_batches}, Loss: {nonfinite_loss_batches}, Grad: {nonfinite_grad_batches}).")

if __name__ == "__main__":
    main()
