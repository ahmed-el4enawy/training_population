"""
train_population_model.py  (fixed version)

Trains the population-level NN state-space model T1DSim_NN^P
(CGMOHSUSimStateSpaceModel_V2) on the merged simulated dataset produced by
merge_parts.py (population_development_dataset_merged.mat), following
Section 2.2.2 of Roquemen-Echeverri et al. (arXiv:2508.05705).
"""

import os
import time
import json
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pickle import dump, load
import h5py
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing._data import _handle_zeros_in_scale

from t1dsim_ai.population_model import CGMOHSUSimStateSpaceModel_V2
from t1dsim_ai.options import n_neurons_pop
from t1dsim_ai.utils.preprocess import scale_single_state, scale_inverse_Q1

# ==============================================================================
# CONFIG - EDIT THESE
# ==============================================================================
MERGED_MAT_PATH = os.environ.get("MERGED_MAT_PATH", "/tmp/cugp012/population_development_dataset_merged.mat")
CACHE_DIR = os.environ.get("CACHE_DIR", "/tmp/cugp012/cache")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/nfs/slurm/cugp012/training_population/models/PopulationModel_v2/")
MODEL_FILENAME = "population_model_trained.pt"
RESUME_CHECKPOINT = os.environ.get("RESUME_CHECKPOINT", None)
BENCHMARK_ONLY = os.environ.get("BENCHMARK_ONLY", "0") == "1"
PREPARE_CACHE_ONLY = os.environ.get("PREPARE_CACHE_ONLY", "0") == "1"

# [ENGINEERING / REPRODUCIBILITY]
SEED = 0

# [PAPER-EXPLICIT]
SEQ_LEN = 61
TRAIN_OVERLAP = 0.75
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 0.0 # [OFFICIAL-CODE-DERIVED]
ALPHA = 0.7
BETA = 0.08
LR_DECAY_PER_EPOCH = np.exp(-0.1)

MAX_EPOCHS = 15
PATIENCE = None
VAL_OVERLAP = 0.0
CHECKPOINT_POLICY = "final_epoch"

MAX_TRAIN_SCENARIOS = None
MAX_VAL_SCENARIOS = None
MAX_TRAIN_BATCHES_PER_EPOCH = None
MAX_VAL_BATCHES_PER_EPOCH = None
SCALER_FIT_ROWS = None
CHUNK_SCENARIOS = 2000

INPUT_COLUMN_PERM = None
STATE_ORDER = ["Q1", "Q2", "S1", "S2", "I", "X1", "X2", "X3", "C2", "C1"]
C1_INDEX = STATE_ORDER.index("C1")

STATE_WEIGHTS = np.array(
    [5/24, 1/6, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/24],
    dtype=np.float32,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_VDG = 0.16
_MGDL_PER_MMOL = 18.0

# Dataset specific known values
EXPECTED_SHA256 = "6FCE64E2D5C695EA61BE5C376C061A6B937D5728B94FE293803D9D83766DA91C"
EXPECTED_DATASET_SIZE = 29312941019

def hash_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()

# ==============================================================================
# Data loading
# ==============================================================================
def get_dataset_dims(path):
    with h5py.File(path, "r") as f:
        n_states, T, n_scenarios = f["dataset_states"].shape
        n_inputs = f["dataset_inputs"].shape[0]
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

def fill_split_buffer_memmap(path, idx, T, n_states, n_inputs, prefix, chunk_size=2000):
    idx = np.asarray(idx)
    total_rows = len(idx) * T
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    x_path = os.path.join(CACHE_DIR, f"{prefix}_states.dat")
    u_path = os.path.join(CACHE_DIR, f"{prefix}_inputs.dat")
    
    x_flat = np.memmap(x_path, dtype=np.float32, mode='w+', shape=(total_rows, n_states))
    u_flat = np.memmap(u_path, dtype=np.float32, mode='w+', shape=(total_rows, n_inputs))
    
    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    pos = 0
    total_scenarios = len(idx)
    scenarios_loaded = 0
    
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
                scenarios_loaded += c
                pct = int(scenarios_loaded / total_scenarios * 100)
                if pct % 5 == 0 and scenarios_loaded == c or (scenarios_loaded % (chunk_size*5) < chunk_size):
                    print(f"{prefix} cache: {pct}%", flush=True)

    x_flat.flush()
    u_flat.flush()
    return x_flat, u_flat

def fit_robust_scaler_memmap(memmap_array):
    n_rows, n_features = memmap_array.shape
    centers = np.zeros(n_features, dtype=np.float64)
    scales = np.zeros(n_features, dtype=np.float64)
    
    for i in range(n_features):
        col = memmap_array[:, i]
        q25, median, q75 = np.percentile(col, [25.0, 50.0, 75.0])
        iqr = q75 - q25
        centers[i] = median
        scales[i] = iqr
        
    scales = _handle_zeros_in_scale(scales, copy=False)
    
    scaler = RobustScaler()
    scaler.center_ = centers
    scaler.scale_ = scales
    scaler.n_features_in_ = n_features
    return scaler

def compute_state_min_memmap(memmap_array, n_states, chunk_rows=500_000):
    n = memmap_array.shape[0]
    global_min = np.full(n_states, np.inf, dtype=np.float32)
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        chunk_min = memmap_array[start:end].min(axis=0)
        global_min = np.minimum(global_min, chunk_min)
    return global_min

def transform_in_place_memmap(scaler, memmap_array, prefix, name, chunk_rows=500_000):
    n = memmap_array.shape[0]
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        memmap_array[start:end] = scaler.transform(memmap_array[start:end]).astype(np.float32)
        pct = int(end / n * 100)
        if (start // chunk_rows) % max(1, (n // chunk_rows // 20)) == 0:
            print(f"Scaling {prefix} {name}: {pct}%", flush=True)
    memmap_array.flush()
    print(f"Scaling {prefix} {name}: 100%", flush=True)

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
        if self.cgm_min is not None and self.cgm_max is not None:
            return torch.clamp(x, min=self.cgm_min, max=self.cgm_max)
        return x

    def forward(self, x0_batch, u_batch):
        X_sim_list = [x0_batch]
        x_step = x0_batch
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
        y_sim = x_sim[1:, :, [0]]
        y_true = x_true[1:, :, [0]]
        penalty = self.fit_penalty(y_sim, y_true, lim_inferior_scaled, lim_superior_scaled)
        L_fit = torch.mean((y_sim - y_true) ** 2 * penalty)

        err = x_sim[1:] - x_true[1:]
        mse_per_state = torch.mean(err ** 2, dim=(0, 1))

        mask = torch.ones_like(mse_per_state)
        mask[self.c1_index] = 0.0
        mse_per_state = mse_per_state * mask

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
    if MAX_EPOCHS is None:
        raise ValueError("MAX_EPOCHS must not be None.")
    if CHECKPOINT_POLICY not in ["best_validation", "final_epoch"]:
        raise ValueError("CHECKPOINT_POLICY must be 'best_validation' or 'final_epoch'.")
    if VAL_OVERLAP is None:
        raise ValueError("VAL_OVERLAP must not be None.")
    if INPUT_COLUMN_PERM is not None:
        raise NotImplementedError("INPUT_COLUMN_PERM not implemented for memmap refactor.")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

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

    manifest_path = os.path.join(CACHE_DIR, "cache_manifest.json")
    cache_valid = False
    
    # Precompute expected sizes
    sz_train_st = len(train_idx) * T * n_states * 4
    sz_train_in = len(train_idx) * T * n_inputs * 4
    sz_val_st = len(val_idx) * T * n_states * 4
    sz_val_in = len(val_idx) * T * n_inputs * 4
    
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
                
            # 1. Dataset size and identity must match
            ds_size = os.path.getsize(MERGED_MAT_PATH)
            if ds_size != EXPECTED_DATASET_SIZE or manifest.get("dataset_size") != EXPECTED_DATASET_SIZE:
                raise ValueError("Dataset size mismatch")
            if manifest.get("dataset_sha256") != EXPECTED_SHA256:
                raise ValueError("Dataset SHA256 mismatch")
            if manifest.get("dtype") != "float32":
                raise ValueError("Cache dtype mismatch")
                
            # 2. Metadata checks
            if not (manifest.get("CACHE_SCHEMA_VERSION") == 1 and
                    manifest.get("train_count") == len(train_idx) and
                    manifest.get("val_count") == len(val_idx) and
                    manifest.get("T") == T and
                    manifest.get("state_count") == n_states and
                    manifest.get("input_count") == n_inputs and
                    manifest.get("STATE_ORDER") == STATE_ORDER and
                    manifest.get("INPUT_COLUMN_PERM") == INPUT_COLUMN_PERM and
                    manifest.get("SCALER_FIT_ROWS") == SCALER_FIT_ROWS):
                raise ValueError("Metadata mismatch")
                
            # 3. File sizes strictly verified
            cache_files = {
                "TRAIN_states.dat": sz_train_st,
                "TRAIN_inputs.dat": sz_train_in,
                "VAL_states.dat": sz_val_st,
                "VAL_inputs.dat": sz_val_in
            }
            for fname, exp_sz in cache_files.items():
                fpath = os.path.join(CACHE_DIR, fname)
                if not os.path.exists(fpath) or os.path.getsize(fpath) != exp_sz:
                    raise ValueError(f"Cache file missing or size mismatch: {fname}")
                    
            scaler_states_path = os.path.join(OUTPUT_DIR, "scaler_states.pkl")
            scaler_inputs_path = os.path.join(OUTPUT_DIR, "scaler_inputs.pkl")
            if not os.path.exists(scaler_states_path) or not os.path.exists(scaler_inputs_path):
                raise ValueError("Scaler files missing")
                
            man_ss = manifest.get("scaler_states", {})
            man_si = manifest.get("scaler_inputs", {})
            
            if hash_file_sha256(scaler_states_path) != man_ss.get("hash"):
                raise ValueError("Scaler states hash mismatch")
            if hash_file_sha256(scaler_inputs_path) != man_si.get("hash"):
                raise ValueError("Scaler inputs hash mismatch")
                
            # 4. Load scalers and strictly compare against manifest metadata
            with open(scaler_states_path, "rb") as fh:
                test_ss = load(fh)
            with open(scaler_inputs_path, "rb") as fh:
                test_si = load(fh)
            
            if not (test_ss.n_features_in_ == man_ss.get("n_features_in_") and
                    np.allclose(test_ss.center_, man_ss.get("center_", [])) and
                    np.allclose(test_ss.scale_, man_ss.get("scale_", [])) and
                    test_si.n_features_in_ == man_si.get("n_features_in_") and
                    np.allclose(test_si.center_, man_si.get("center_", [])) and
                    np.allclose(test_si.scale_, man_si.get("scale_", []))):
                raise ValueError("Scaler metadata mismatch")
                
            cache_valid = True
            state_min = np.array(manifest["state_min"], dtype=np.float32)
            if len(state_min) != n_states or not np.isfinite(state_min).all():
                raise ValueError("state_min is invalid")
            scaler_states = test_ss
            scaler_inputs = test_si
            
        except Exception as e:
            print(f"Cache validation failed: {e}. Rebuilding...")
            cache_valid = False

    if not cache_valid:
        if BENCHMARK_ONLY:
            raise RuntimeError("Benchmark mode requires a valid completed cache. Run PREPARE_CACHE_ONLY=1 first.")

        print("No valid cache found. Rebuilding cache...")
        
        # Verify dataset size and SHA-256 before building
        actual_size = os.path.getsize(MERGED_MAT_PATH)
        if actual_size != EXPECTED_DATASET_SIZE:
            raise RuntimeError(f"Dataset size mismatch! Expected {EXPECTED_DATASET_SIZE}, got {actual_size}")
            
        print("Verifying dataset SHA-256 (this may take a minute)...")
        t_hash = time.time()
        actual_sha256 = hash_file_sha256(MERGED_MAT_PATH)
        if actual_sha256 != EXPECTED_SHA256:
            raise RuntimeError(f"Dataset SHA-256 mismatch! Expected {EXPECTED_SHA256}, got {actual_sha256}")
        print(f"Dataset verified successfully [{time.time() - t_hash:.1f}s].")

        t0 = time.time()
        x_train_flat, u_train_flat = fill_split_buffer_memmap(MERGED_MAT_PATH, train_idx, T, n_states, n_inputs, prefix="TRAIN", chunk_size=CHUNK_SCENARIOS)
        print(f"  done loading TRAIN scenarios in {time.time()-t0:.1f}s")
        
        print("Fitting RobustScaler on TRAIN ...")
        scaler_states = fit_robust_scaler_memmap(x_train_flat)
        scaler_inputs = fit_robust_scaler_memmap(u_train_flat)
        
        scaler_states_path = os.path.join(OUTPUT_DIR, "scaler_states.pkl")
        scaler_inputs_path = os.path.join(OUTPUT_DIR, "scaler_inputs.pkl")
        
        tmp_ss = scaler_states_path + ".tmp"
        with open(tmp_ss, "wb") as fh:
            dump(scaler_states, fh)
        os.replace(tmp_ss, scaler_states_path)
        
        tmp_si = scaler_inputs_path + ".tmp"
        with open(tmp_si, "wb") as fh:
            dump(scaler_inputs, fh)
        os.replace(tmp_si, scaler_inputs_path)
        
        ss_sha256 = hash_file_sha256(scaler_states_path)
        si_sha256 = hash_file_sha256(scaler_inputs_path)
        print(f"  saved scalers atomically to {OUTPUT_DIR}")
        
        print("Scaling TRAIN in place ...")
        transform_in_place_memmap(scaler_states, x_train_flat, "TRAIN", "states")
        transform_in_place_memmap(scaler_inputs, u_train_flat, "TRAIN", "inputs")
        
        print(f"Loading VAL scenarios ...")
        x_val_flat, u_val_flat = fill_split_buffer_memmap(MERGED_MAT_PATH, val_idx, T, n_states, n_inputs, prefix="VAL", chunk_size=CHUNK_SCENARIOS)
        
        print("Scaling VAL in place ...")
        transform_in_place_memmap(scaler_states, x_val_flat, "VAL", "states")
        transform_in_place_memmap(scaler_inputs, u_val_flat, "VAL", "inputs")
        
        print("Computing state min...")
        state_min = compute_state_min_memmap(x_train_flat, n_states)
        state_min[C1_INDEX] = (0.0 - scaler_states.center_[C1_INDEX]) / scaler_states.scale_[C1_INDEX]
        
        manifest = {
            "CACHE_SCHEMA_VERSION": 1,
            "dataset_sha256": actual_sha256,
            "dataset_size": actual_size,
            "train_count": len(train_idx),
            "val_count": len(val_idx),
            "T": T,
            "state_count": n_states,
            "input_count": n_inputs,
            "STATE_ORDER": STATE_ORDER,
            "INPUT_COLUMN_PERM": INPUT_COLUMN_PERM,
            "SCALER_FIT_ROWS": SCALER_FIT_ROWS,
            "dtype": "float32",
            "state_min": state_min.tolist(),
            "scaler_states": {
                "hash": ss_sha256,
                "n_features_in_": scaler_states.n_features_in_,
                "center_": scaler_states.center_.tolist(),
                "scale_": scaler_states.scale_.tolist()
            },
            "scaler_inputs": {
                "hash": si_sha256,
                "n_features_in_": scaler_inputs.n_features_in_,
                "center_": scaler_inputs.center_.tolist(),
                "scale_": scaler_inputs.scale_.tolist()
            }
        }
        tmp_manifest = manifest_path + ".tmp"
        with open(tmp_manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_manifest, manifest_path)
        print("Cache built and manifest saved.")
    else:
        print("Valid cache found. Reusing memmaps...")
        x_train_flat = np.memmap(os.path.join(CACHE_DIR, "TRAIN_states.dat"), dtype=np.float32, mode='r', shape=(len(train_idx)*T, n_states))
        u_train_flat = np.memmap(os.path.join(CACHE_DIR, "TRAIN_inputs.dat"), dtype=np.float32, mode='r', shape=(len(train_idx)*T, n_inputs))
        x_val_flat = np.memmap(os.path.join(CACHE_DIR, "VAL_states.dat"), dtype=np.float32, mode='r', shape=(len(val_idx)*T, n_states))
        u_val_flat = np.memmap(os.path.join(CACHE_DIR, "VAL_inputs.dat"), dtype=np.float32, mode='r', shape=(len(val_idx)*T, n_inputs))

    if PREPARE_CACHE_ONLY:
        print("PREPARE_CACHE_ONLY finished. Cache is valid and ready. Exiting.")
        return

    x_train = x_train_flat.reshape(len(train_idx), T, n_states)
    u_train = u_train_flat.reshape(len(train_idx), T, n_inputs)
    x_val = x_val_flat.reshape(len(val_idx), T, n_states)
    u_val = u_val_flat.reshape(len(val_idx), T, n_inputs)

    lim_inferior_scaled = scale_single_state(70, "Q1", OUTPUT_DIR)
    lim_superior_scaled = scale_single_state(250, "Q1", OUTPUT_DIR)
    cgm_min_scaled = None
    cgm_max_scaled = None

    print(f"Building model on device: {DEVICE}")
    model = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
    model.to(DEVICE)

    simulator = ForwardEulerSimulatorPop(model, ts=5.0, cgm_min=cgm_min_scaled, cgm_max=cgm_max_scaled).to(DEVICE)
    loss_fn = PopulationLoss(STATE_WEIGHTS, state_min, C1_INDEX, alpha=ALPHA, beta=BETA).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY_PER_EPOCH)

    train_sampler = FramedWindowSampler(x_train, u_train, SEQ_LEN, TRAIN_OVERLAP, BATCH_SIZE, DEVICE, seed=SEED)
    val_sampler = FramedWindowSampler(x_val, u_val, SEQ_LEN, VAL_OVERLAP, BATCH_SIZE, DEVICE, seed=SEED + 1)
    
    iters_per_epoch = train_sampler.n_batches_per_epoch()
    if MAX_TRAIN_BATCHES_PER_EPOCH is not None:
        iters_per_epoch = min(iters_per_epoch, MAX_TRAIN_BATCHES_PER_EPOCH)
        
    val_batches_per_epoch = val_sampler.n_batches_per_epoch()
    if MAX_VAL_BATCHES_PER_EPOCH is not None:
        val_batches_per_epoch = min(val_batches_per_epoch, MAX_VAL_BATCHES_PER_EPOCH)
    
    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if RESUME_CHECKPOINT:
        print(f"Resuming from checkpoint: {RESUME_CHECKPOINT}")
        ckpt = torch.load(RESUME_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['completed_epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        epochs_without_improvement = ckpt['epochs_without_improvement']
        
        np.random.set_state(ckpt['numpy_rng_state'])
        torch.set_rng_state(ckpt['torch_cpu_rng_state'])
        if DEVICE.type == "cuda":
            torch.cuda.set_rng_state(ckpt['torch_cuda_rng_state'])
            
        train_sampler.rng.set_state(ckpt['train_sampler_rng_state'])
        train_sampler.epoch_order = ckpt['train_sampler_epoch_order']
        train_sampler.pos = ckpt['train_sampler_pos']
        print(f"Resumed successfully. Continuing from epoch {start_epoch}")

    if BENCHMARK_ONLY:
        BENCHMARK_BATCHES = 50
        print(f"\n--- RUNNING BENCHMARK_ONLY ({BENCHMARK_BATCHES} actual training batches) ---")
        model.train()
        
        bm_model_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        bm_opt_sd = optimizer.state_dict()
        bm_np_rng = np.random.get_state()
        bm_torch_rng = torch.get_rng_state()
        if DEVICE.type == "cuda": bm_cuda_rng = torch.cuda.get_rng_state()
        bm_sampler_rng = train_sampler.rng.get_state()
        bm_sampler_order = train_sampler.epoch_order.copy()
        bm_sampler_pos = train_sampler.pos
        
        for _ in range(5):
            x0, u_batch, x_true = train_sampler.next_batch()
            x_sim = simulator(x0, u_batch)
            loss, _, _ = loss_fn(x_sim, x_true, lim_inferior_scaled, lim_superior_scaled)
            loss.backward()
            optimizer.zero_grad()
            
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t_bm_start = time.time()
        
        for _ in range(BENCHMARK_BATCHES):
            x0, u_batch, x_true = train_sampler.next_batch()
            x_sim = simulator(x0, u_batch)
            loss, _, _ = loss_fn(x_sim, x_true, lim_inferior_scaled, lim_superior_scaled)
            loss.backward()
            # NO OPTIMIZER STEP!
            optimizer.zero_grad()
            
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t_bm_end = time.time()
        
        sec_per_batch = (t_bm_end - t_bm_start) / BENCHMARK_BATCHES
        est_train_epoch = sec_per_batch * iters_per_epoch
        
        print("GPU name:", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU")
        print("Number of timed batches:", BENCHMARK_BATCHES)
        print(f"Seconds per real training batch: {sec_per_batch:.3f} s")
        print(f"Estimated training-only time per epoch: {est_train_epoch/3600:.2f} hours")
        print("Training batches per epoch:", iters_per_epoch)
        print("Validation batches per epoch:", val_batches_per_epoch)
        print(f"Rough estimated 15-epoch training-only time: {(est_train_epoch * 15)/3600:.2f} hours")
        
        model.load_state_dict(bm_model_sd)
        optimizer.load_state_dict(bm_opt_sd)
        np.random.set_state(bm_np_rng)
        torch.set_rng_state(bm_torch_rng)
        if DEVICE.type == "cuda": torch.cuda.set_rng_state(bm_cuda_rng)
        train_sampler.rng.set_state(bm_sampler_rng)
        train_sampler.epoch_order = bm_sampler_order
        train_sampler.pos = bm_sampler_pos
        print("BENCHMARK_ONLY finished. State restored. Exiting.")
        return

    val_fixed_batches = val_sampler.get_fixed_subset(val_batches_per_epoch)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train()
        epoch_losses = []
        t_epoch = time.time()
        print_every = max(1, min(50, iters_per_epoch // 10))

        for it in range(iters_per_epoch):
            optimizer.zero_grad()
            x0, u_batch, x_true = train_sampler.next_batch()
            x_sim = simulator(x0, u_batch)

            if not torch.isfinite(x_sim).all():
                raise RuntimeError("Non-finite simulation detected during final training.")

            loss, _, _ = loss_fn(x_sim, x_true, lim_inferior_scaled, lim_superior_scaled)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss detected during final training.")
                
            loss.backward()
            
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise RuntimeError("Non-finite gradient detected during final training.")
                
            optimizer.step()
            epoch_losses.append(loss.item())

            if (it + 1) % print_every == 0 or (it + 1) == iters_per_epoch:
                running_mean = np.mean(epoch_losses) if epoch_losses else float("nan")
                elapsed = time.time() - t_epoch
                eta = elapsed / (it + 1) * iters_per_epoch
                print(f"  epoch {epoch} [train {it+1}/{iters_per_epoch}] "
                      f"loss {running_mean:.6f} | elapsed {elapsed:.1f}s | estimated epoch training time {eta:.1f}s")

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
        else:
            epochs_without_improvement += 1
            if PATIENCE is not None and epochs_without_improvement >= PATIENCE:
                print(f"Early stopping after {epochs_without_improvement} epochs.")
                break
                
        ckpt = {
            'completed_epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'epochs_without_improvement': epochs_without_improvement,
            'numpy_rng_state': np.random.get_state(),
            'torch_cpu_rng_state': torch.get_rng_state(),
            'torch_cuda_rng_state': torch.cuda.get_rng_state() if DEVICE.type == "cuda" else None,
            'train_sampler_rng_state': train_sampler.rng.get_state(),
            'train_sampler_epoch_order': train_sampler.epoch_order,
            'train_sampler_pos': train_sampler.pos
        }
        ckpt_path = os.path.join(OUTPUT_DIR, "training_state_latest.pt")
        tmp_ckpt_path = ckpt_path + ".tmp"
        torch.save(ckpt, tmp_ckpt_path)
        os.replace(tmp_ckpt_path, ckpt_path)
        print(f"  -> saved resumable training state to {ckpt_path}")

        if CHECKPOINT_POLICY == "final_epoch" or (CHECKPOINT_POLICY == "best_validation" and epochs_without_improvement == 0):
            model_out_path = os.path.join(OUTPUT_DIR, MODEL_FILENAME)
            tmp_model_path = model_out_path + ".tmp"
            torch.save(model.state_dict(), tmp_model_path)
            os.replace(tmp_model_path, model_out_path)
            print(f"  -> saved final_epoch checkpoint to {model_out_path}")

    print(f"Training complete.")

if __name__ == "__main__":
    main()
