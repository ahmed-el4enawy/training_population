"""
train_population_model.py

Trains the population-level neural state-space model T1DSim_NN^P
(CGMOHSUSimStateSpaceModel_V2 / PackedPopulationModel) on the reproduced
OHSU glucoregulatory population dataset (SIM002, SIM004, SIM005),
faithfully following Section 2.2.2 of Roquemen-Echeverri et al. (arXiv:2508.05705).
"""

import os
import sys
import time
import json
import hashlib
import copy
import argparse
from pathlib import Path
from pickle import dump, load
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing._data import _handle_zeros_in_scale

# Submodule & package imports
REPO_ROOT = Path(__file__).resolve().parent
T1DSIM_AI_SRC = REPO_ROOT / "T1DSim_AI" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(T1DSIM_AI_SRC) not in sys.path:
    sys.path.insert(0, str(T1DSIM_AI_SRC))

from t1dsim_ai.population_model import CGMOHSUSimStateSpaceModel_V2
from t1dsim_ai.options import n_neurons_pop
from t1dsim_ai.utils.preprocess import scale_inverse_Q1
from packed_population_model import PackedPopulationModel
from dataset_pipeline import (
    STATE_ORDER,
    INPUT_ORDER,
    TRAJECTORY_LENGTH,
    DEFAULT_DATASET_DIR,
    DEFAULT_CACHE_DIR,
    build_memmap_cache,
    validate_cache,
    load_cached_split,
)

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS (PAPER-EXPLICIT & OFFICIAL ARTIFACTS)
# ==============================================================================
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models" / "PopulationModel_v2"
MODEL_FILENAME = "population_model_trained.pt"

SEED = 0
SEQ_LEN = 61                  # 5 hours at ts=5 min
TRAIN_OVERLAP = 0.75          # 75% sliding candidate overlap
BATCH_SIZE = 128              # Batch size
LR = 1e-3                     # Initial learning rate
WEIGHT_DECAY = 0.0            # No weight decay
ALPHA = 0.7                   # State consistency weight (Eq. 9)
BETA = 0.08                   # Soft barrier constraint weight (Eq. 7)
LR_DECAY_PER_EPOCH = np.exp(-0.1)  # Exponential decay per epoch (~0.9048)

MAX_EPOCHS = 15               # Exact official epoch count
VAL_OVERLAP = 0.0             # Validation stride (no overlap)
CHECKPOINT_POLICY = "final_epoch"
TRAIN_SEQUENCE_LIMIT = 1_000_000  # Paper reports ~1 million 5-hour sequences

C1_INDEX = STATE_ORDER.index("C1")  # Index 9
STATE_WEIGHTS = np.array(
    [5/24, 1/6, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/24],
    dtype=np.float32,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_VDG = 0.16
_MGDL_PER_MMOL = 18.0


def hash_file_sha256(path: str, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest().upper()


def fit_robust_scaler_memmap(memmap_array: np.ndarray) -> RobustScaler:
    """Fit RobustScaler on 2D memmap without loading entire array into RAM at once."""
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


def compute_state_min_memmap(memmap_array: np.ndarray, n_states: int, chunk_rows: int = 500_000) -> np.ndarray:
    """Compute per-feature minimum across memory-mapped rows."""
    n = memmap_array.shape[0]
    global_min = np.full(n_states, np.inf, dtype=np.float32)
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        chunk_min = memmap_array[start:end].min(axis=0)
        global_min = np.minimum(global_min, chunk_min)
    return global_min


def transform_in_place_memmap(scaler: RobustScaler, memmap_array: np.ndarray, prefix: str, name: str, chunk_rows: int = 500_000):
    """Apply scaler transform in chunks directly modifying the memory-mapped file."""
    n = memmap_array.shape[0]
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        memmap_array[start:end] = scaler.transform(memmap_array[start:end]).astype(np.float32)
        pct = int(end / n * 100)
        if (start // chunk_rows) % max(1, (n // chunk_rows // 10)) == 0:
            print(f"  Scaling {prefix} {name}: {pct}%", flush=True)
    memmap_array.flush()
    print(f"  Scaling {prefix} {name}: 100%", flush=True)


# ==============================================================================
# Framed Batch Sampler (true 75% overlap, index pairs, boundary safe)
# ==============================================================================
class FramedWindowSampler:
    def __init__(
        self,
        x_est: np.ndarray,
        u_fit: np.ndarray,
        seq_len: int,
        overlap: float,
        batch_size: int,
        device: torch.device,
        seed: int = 0,
        max_sequences: Optional[int] = None,
    ):
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
        if max_sequences is not None and len(self.pairs) > max_sequences:
            chosen = self.rng.choice(len(self.pairs), size=int(max_sequences), replace=False)
            self.pairs = self.pairs[chosen]

        self._reshuffle()

    def _reshuffle(self):
        self.epoch_order = self.rng.permutation(len(self.pairs))
        self.pos = 0

    def n_batches_per_epoch(self) -> int:
        return len(self.pairs) // self.batch_size

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.pos + self.batch_size > len(self.pairs):
            self._reshuffle()
        chosen = self.epoch_order[self.pos : self.pos + self.batch_size]
        self.pos += self.batch_size
        return self.batch_from_pairs(self.pairs[chosen])

    def get_fixed_subset(self, n_batches: int) -> List[np.ndarray]:
        n_needed = min(n_batches * self.batch_size, len(self.pairs))
        chosen = self.rng.permutation(len(self.pairs))[:n_needed]
        return [
            self.pairs[chosen[s : s + self.batch_size]]
            for s in range(0, len(chosen), self.batch_size)
        ]

    def batch_from_pairs(self, pair_batch: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
# Forward Euler Simulator
# ==============================================================================
class ForwardEulerSimulatorPop(nn.Module):
    def __init__(self, ss_pop_model, ts: float = 5.0, cgm_min: Optional[float] = None, cgm_max: Optional[float] = None):
        super().__init__()
        self.ss_pop_model = ss_pop_model
        self.ts = ts
        self.cgm_min = cgm_min
        self.cgm_max = cgm_max

    def adjust_cgm(self, x: torch.Tensor) -> torch.Tensor:
        if self.cgm_min is not None and self.cgm_max is not None:
            return torch.clamp(x, min=self.cgm_min, max=self.cgm_max)
        return x

    def forward(self, x0_batch: torch.Tensor, u_batch: torch.Tensor) -> torch.Tensor:
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
# Population Loss (Equations 5-9)
# ==============================================================================
class PopulationLoss:
    def __init__(self, state_weights, state_min, c1_index: int, alpha: float = 0.7, beta: float = 0.08):
        self.w = torch.tensor(state_weights, dtype=torch.float32)
        self.state_min = torch.tensor(state_min, dtype=torch.float32)
        self.c1_index = c1_index
        self.alpha = alpha
        self.beta = beta

    def to(self, device: torch.device):
        self.w = self.w.to(device)
        self.state_min = self.state_min.to(device)
        return self

    @staticmethod
    def fit_penalty(y_sim: torch.Tensor, y_true: torch.Tensor, lim_inferior: float, lim_superior: float) -> torch.Tensor:
        penalty = torch.ones_like(y_true)
        penalty[torch.logical_and(y_true <= lim_inferior, y_sim > y_true)] = 2.0
        penalty[torch.logical_and(y_true >= lim_superior, y_sim < y_true)] = 6.0
        return penalty

    def __call__(self, x_sim: torch.Tensor, x_true: torch.Tensor, lim_inferior_scaled: float, lim_superior_scaled: float):
        y_sim = x_sim[1:, :, [0]]
        y_true = x_true[1:, :, [0]]
        penalty = self.fit_penalty(y_sim, y_true, lim_inferior_scaled, lim_superior_scaled)
        L_fit = torch.mean((y_sim - y_true) ** 2 * penalty)

        err = x_sim[1:] - x_true[1:]
        mse_per_state = torch.mean(err ** 2, dim=(0, 1))

        # State C1 (state_Ug2 = 0) is excluded from consistency MSE penalty
        mask = torch.ones_like(mse_per_state)
        mask[self.c1_index] = 0.0
        mse_per_state = mse_per_state * mask

        floor = self.state_min
        soft_constraint = torch.clamp(-(x_sim[1:] - floor), min=0.0)
        soft_penalty_per_state = torch.sum(soft_constraint, dim=(0, 1))

        per_state_loss = mse_per_state + self.beta * soft_penalty_per_state
        L_consistency = torch.sum(self.w * per_state_loss)

        L_total = L_fit + self.alpha * L_consistency
        return L_total, L_fit, L_consistency


# ==============================================================================
# Primary Training Routine
# ==============================================================================
def train(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    resume_checkpoint: Optional[str] = None,
    benchmark_only: bool = False,
    prepare_cache_only: bool = False,
    use_packed_model: bool = True,
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    train_sequence_limit: Optional[int] = TRAIN_SEQUENCE_LIMIT,
    run_validation: bool = False,
    use_reference_scalers: bool = False,
    force_cache_rebuild: bool = False,
):
    dataset_dir = Path(dataset_dir).resolve()
    cache_dir = Path(cache_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("T1DSim Glucoregulatory Population Neural Model Training")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})")
    print(f"Dataset root: {dataset_dir}")
    print(f"Cache dir:    {cache_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Architecture: {'PackedPopulationModel (CUDA-optimized equivalent)' if use_packed_model else 'Official CGMOHSUSimStateSpaceModel_V2'}")
    print("=" * 70)

    # 1. Dataset cache validation & ingestion
    valid, manifest = validate_cache(cache_dir)
    if not valid or force_cache_rebuild:
        print("Building binary memory-mapped cache from .pkl dataset...")
        manifest = build_memmap_cache(dataset_dir, cache_dir, force_rebuild=True)
    else:
        print(f"Loaded verified cache manifest (Schema v{manifest.get('CACHE_SCHEMA_VERSION')})")

    # 2. Scaler handling
    scaler_states_path = output_dir / "scaler_states.pkl"
    scaler_inputs_path = output_dir / "scaler_inputs.pkl"

    if use_reference_scalers:
        ref_ss_path = REPO_ROOT / "models" / "PopulationModel" / "scaler_states.pkl"
        ref_si_path = REPO_ROOT / "models" / "PopulationModel" / "scaler_inputs.pkl"
        print(f"Loading reference scalers from {ref_ss_path.parent}...")
        with open(ref_ss_path, "rb") as fh:
            scaler_states = load(fh)
        with open(ref_si_path, "rb") as fh:
            scaler_inputs = load(fh)
    elif scaler_states_path.exists() and scaler_inputs_path.exists():
        print(f"Loading existing fitted scalers from {output_dir}...")
        with open(scaler_states_path, "rb") as fh:
            scaler_states = load(fh)
        with open(scaler_inputs_path, "rb") as fh:
            scaler_inputs = load(fh)
    else:
        print("Fitting RobustScaler on raw TRAIN split...")
        x_raw, u_raw, _ = load_cached_split(cache_dir, "TRAIN", mode="r")
        scaler_states = fit_robust_scaler_memmap(x_raw)
        scaler_inputs = fit_robust_scaler_memmap(u_raw)

        with open(scaler_states_path, "wb") as fh:
            dump(scaler_states, fh)
        with open(scaler_inputs_path, "wb") as fh:
            dump(scaler_inputs, fh)
        print(f"  -> Saved scalers atomically to {output_dir}")

    # 3. Memory-mapped scaled arrays
    # Check if pre-scaled cache exists, otherwise scale in place into scaled dat files
    scaled_train_x_path = cache_dir / "TRAIN_states_scaled.dat"
    scaled_train_u_path = cache_dir / "TRAIN_inputs_scaled.dat"
    scaled_val_x_path = cache_dir / "VAL_states_scaled.dat"
    scaled_val_u_path = cache_dir / "VAL_inputs_scaled.dat"

    train_rows = manifest["shapes"]["TRAIN_states"][0]
    val_rows = manifest["shapes"]["VAL_states"][0]

    if not scaled_train_x_path.exists() or force_cache_rebuild:
        print("Creating pre-scaled TRAIN memory maps...")
        raw_x_train, raw_u_train, _ = load_cached_split(cache_dir, "TRAIN", mode="r")
        x_scaled = np.memmap(scaled_train_x_path, dtype=np.float32, mode="w+", shape=(train_rows, 10))
        u_scaled = np.memmap(scaled_train_u_path, dtype=np.float32, mode="w+", shape=(train_rows, 2))

        # Copy raw and scale in chunks
        chunk = 500_000
        for s in range(0, train_rows, chunk):
            e = min(s + chunk, train_rows)
            x_scaled[s:e] = scaler_states.transform(raw_x_train[s:e]).astype(np.float32)
            u_scaled[s:e] = scaler_inputs.transform(raw_u_train[s:e]).astype(np.float32)
        x_scaled.flush()
        u_scaled.flush()

    if not scaled_val_x_path.exists() or force_cache_rebuild:
        print("Creating pre-scaled VAL memory maps...")
        raw_x_val, raw_u_val, _ = load_cached_split(cache_dir, "VAL", mode="r")
        xv_scaled = np.memmap(scaled_val_x_path, dtype=np.float32, mode="w+", shape=(val_rows, 10))
        uv_scaled = np.memmap(scaled_val_u_path, dtype=np.float32, mode="w+", shape=(val_rows, 2))

        chunk = 500_000
        for s in range(0, val_rows, chunk):
            e = min(s + chunk, val_rows)
            xv_scaled[s:e] = scaler_states.transform(raw_x_val[s:e]).astype(np.float32)
            uv_scaled[s:e] = scaler_inputs.transform(raw_u_val[s:e]).astype(np.float32)
        xv_scaled.flush()
        uv_scaled.flush()

    if prepare_cache_only:
        print("PREPARE_CACHE_ONLY completed successfully. Cache and scalers are ready.")
        return

    # Load scaled memmaps
    x_train_scaled = np.memmap(scaled_train_x_path, dtype=np.float32, mode="r", shape=(train_rows, 10))
    u_train_scaled = np.memmap(scaled_train_u_path, dtype=np.float32, mode="r", shape=(train_rows, 2))
    x_val_scaled = np.memmap(scaled_val_x_path, dtype=np.float32, mode="r", shape=(val_rows, 10))
    u_val_scaled = np.memmap(scaled_val_u_path, dtype=np.float32, mode="r", shape=(val_rows, 2))

    n_train_traj = train_rows // TRAJECTORY_LENGTH
    n_val_traj = val_rows // TRAJECTORY_LENGTH

    x_train_3d = x_train_scaled.reshape(n_train_traj, TRAJECTORY_LENGTH, 10)
    u_train_3d = u_train_scaled.reshape(n_train_traj, TRAJECTORY_LENGTH, 2)
    x_val_3d = x_val_scaled.reshape(n_val_traj, TRAJECTORY_LENGTH, 10)
    u_val_3d = u_val_scaled.reshape(n_val_traj, TRAJECTORY_LENGTH, 2)

    print(f"\nTraining set:   {n_train_traj:,} trajectories ({train_rows:,} timesteps)")
    print(f"Validation set: {n_val_traj:,} trajectories ({val_rows:,} timesteps)")

    # Compute physical boundaries (Eq. 7 clinical thresholds: hypo < 70, hyper > 250 mg/dL)
    lim_inferior_scaled = float((70.0 - scaler_states.center_[0]) / scaler_states.scale_[0])
    lim_superior_scaled = float((250.0 - scaler_states.center_[0]) / scaler_states.scale_[0])
    cgm_min_scaled = float((39.0 - scaler_states.center_[0]) / scaler_states.scale_[0])
    cgm_max_scaled = float((401.0 - scaler_states.center_[0]) / scaler_states.scale_[0])

    state_min = compute_state_min_memmap(x_train_scaled, 10)
    state_min[C1_INDEX] = (0.0 - scaler_states.center_[C1_INDEX]) / scaler_states.scale_[C1_INDEX]

    # Initialize model
    official_init_model = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
    if use_packed_model:
        model = PackedPopulationModel(official_init_model, n_neurons_pop)
        print(f"  Initialized PackedPopulationModel: {model.active_parameter_count():,} active parameters")
    else:
        model = official_init_model
        print(f"  Initialized CGMOHSUSimStateSpaceModel_V2: {sum(p.numel() for p in model.parameters()):,} parameters")

    model.to(DEVICE)
    simulator = ForwardEulerSimulatorPop(model, ts=5.0, cgm_min=cgm_min_scaled, cgm_max=cgm_max_scaled).to(DEVICE)
    loss_fn = PopulationLoss(STATE_WEIGHTS, state_min, C1_INDEX, alpha=ALPHA, beta=BETA).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY_PER_EPOCH)

    train_sampler = FramedWindowSampler(
        x_train_3d, u_train_3d, SEQ_LEN, TRAIN_OVERLAP, batch_size, DEVICE,
        seed=SEED, max_sequences=train_sequence_limit
    )
    val_sampler = FramedWindowSampler(
        x_val_3d, u_val_3d, SEQ_LEN, VAL_OVERLAP, batch_size, DEVICE, seed=SEED + 1
    )

    iters_per_epoch = train_sampler.n_batches_per_epoch()
    val_batches_per_epoch = val_sampler.n_batches_per_epoch()
    print(f"Sequence pool: {len(train_sampler.pairs):,} sequences -> {iters_per_epoch:,} batches/epoch (B={batch_size})")

    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["completed_epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]

        np.random.set_state(ckpt["numpy_rng_state"])
        torch.set_rng_state(ckpt["torch_cpu_rng_state"])
        if DEVICE.type == "cuda":
            torch.cuda.set_rng_state(ckpt["torch_cuda_rng_state"])

        train_sampler.rng.set_state(ckpt["train_sampler_rng_state"])
        train_sampler.epoch_order = ckpt["train_sampler_epoch_order"]
        train_sampler.pos = ckpt["train_sampler_pos"]
        print(f"Resumed successfully. Continuing from epoch {start_epoch}")

    if benchmark_only:
        BENCHMARK_BATCHES = 50
        print(f"\n--- RUNNING BENCHMARK_ONLY ({BENCHMARK_BATCHES} actual training batches) ---")
        model.train()

        bm_model_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        bm_opt_sd = copy.deepcopy(optimizer.state_dict())
        bm_np_rng = np.random.get_state()
        bm_torch_rng = torch.get_rng_state()
        if DEVICE.type == "cuda":
            bm_cuda_rng = torch.cuda.get_rng_state()
        bm_sampler_rng = train_sampler.rng.get_state()
        bm_sampler_order = train_sampler.epoch_order.copy()
        bm_sampler_pos = train_sampler.pos

        # Warm up
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            x0, u_batch, x_true = train_sampler.next_batch()
            x_sim = simulator(x0, u_batch)
            loss, _, _ = loss_fn(x_sim, x_true, lim_inferior_scaled, lim_superior_scaled)
            loss.backward()
            optimizer.step()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_bm_start = time.time()

        for _ in range(BENCHMARK_BATCHES):
            optimizer.zero_grad(set_to_none=True)
            x0, u_batch, x_true = train_sampler.next_batch()
            x_sim = simulator(x0, u_batch)
            loss, _, _ = loss_fn(x_sim, x_true, lim_inferior_scaled, lim_superior_scaled)
            loss.backward()
            optimizer.step()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_bm_end = time.time()

        sec_per_batch = (t_bm_end - t_bm_start) / BENCHMARK_BATCHES
        est_train_epoch = sec_per_batch * iters_per_epoch

        print("GPU name:", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU")
        print("Number of timed batches:", BENCHMARK_BATCHES)
        print(f"Seconds per production batch: {sec_per_batch:.3f} s")
        print(f"Estimated training-only time per epoch: {est_train_epoch / 3600:.2f} hours")
        print("Training batches per epoch:", iters_per_epoch)
        projected_hours = (est_train_epoch * max_epochs) / 3600
        print(f"Projected {max_epochs}-epoch training-only time: {projected_hours:.2f} hours")
        print(f"72-hour target margin: {72.0 - projected_hours:.2f} hours")

        model.load_state_dict(bm_model_sd)
        optimizer.load_state_dict(bm_opt_sd)
        np.random.set_state(bm_np_rng)
        torch.set_rng_state(bm_torch_rng)
        if DEVICE.type == "cuda":
            torch.cuda.set_rng_state(bm_cuda_rng)
        train_sampler.rng.set_state(bm_sampler_rng)
        train_sampler.epoch_order = bm_sampler_order
        train_sampler.pos = bm_sampler_pos
        print("BENCHMARK_ONLY finished. State restored. Exiting.")
        return

    val_fixed_batches = (
        val_sampler.get_fixed_subset(val_batches_per_epoch)
        if run_validation else []
    )

    print("\nStarting model training loop...")
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        epoch_loss_sum = torch.zeros((), dtype=torch.float32, device=DEVICE)
        t_epoch = time.time()
        print_every = max(1, min(100, iters_per_epoch // 10))

        for it in range(iters_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            x0, u_batch, x_true = train_sampler.next_batch()
            x_sim = simulator(x0, u_batch)
            loss, _, _ = loss_fn(x_sim, x_true, lim_inferior_scaled, lim_superior_scaled)
            loss.backward()

            # Synchronized finite check every 100 steps
            if it % 100 == 0:
                grad_tensors = [p.grad.reshape(-1) for p in model.parameters() if p.grad is not None]
                finite = torch.isfinite(x_sim).all() & torch.isfinite(loss)
                if grad_tensors:
                    finite = finite & torch.isfinite(torch.cat(grad_tensors)).all()
                if not bool(finite.item()):
                    raise RuntimeError("Non-finite simulation/loss/gradient detected during training.")

            optimizer.step()
            epoch_loss_sum.add_(loss.detach())

            if (it + 1) % print_every == 0 or (it + 1) == iters_per_epoch:
                running_mean = float((epoch_loss_sum / (it + 1)).item())
                elapsed = time.time() - t_epoch
                eta = elapsed / (it + 1) * iters_per_epoch
                print(
                    f"  epoch {epoch:2d}/{max_epochs} [batch {it+1:5d}/{iters_per_epoch}] "
                    f"loss: {running_mean:.6f} | elapsed: {elapsed:.1f}s | est epoch time: {eta:.1f}s"
                )

        scheduler.step()
        mean_train_loss = float(epoch_loss_sum.item() / max(1, iters_per_epoch))

        if run_validation:
            model.eval()
            with torch.no_grad():
                val_losses = []
                sq_err_sum = 0.0
                n_err_vals = 0
                for v_it, pair_batch in enumerate(val_fixed_batches):
                    x0, u_batch, x_true = val_sampler.batch_from_pairs(pair_batch)
                    x_sim = simulator(x0, u_batch)
                    if not torch.isfinite(x_sim).all():
                        continue

                    loss, _, _ = loss_fn(x_sim, x_true, lim_inferior_scaled, lim_superior_scaled)
                    val_losses.append(loss.item())

                    y_sim_mgdl = scale_inverse_Q1(x_sim[1:, :, [0]].cpu().numpy(), str(output_dir))
                    y_true_mgdl = scale_inverse_Q1(x_true[1:, :, [0]].cpu().numpy(), str(output_dir))
                    batch_sq_err = (y_sim_mgdl - y_true_mgdl) ** 2
                    sq_err_sum += batch_sq_err.sum()
                    n_err_vals += batch_sq_err.size

            mean_val_loss = np.mean(val_losses) if val_losses else float("nan")
            rmse_mgdl = np.sqrt(sq_err_sum / n_err_vals) if n_err_vals > 0 else float("nan")
            print(
                f"Epoch {epoch:4d} | train_loss {mean_train_loss:.6f} | "
                f"val_loss {mean_val_loss:.6f} | val_RMSE {rmse_mgdl:.2f} mg/dL | "
                f"lr {optimizer.param_groups[0]['lr']:.2e} | time {time.time()-t_epoch:.1f}s"
            )

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            print(
                f"Epoch {epoch:4d} | train_loss {mean_train_loss:.6f} | "
                f"lr {optimizer.param_groups[0]['lr']:.2e} | time {time.time()-t_epoch:.1f}s"
            )

        # Save checkpoint
        ckpt = {
            "completed_epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "numpy_rng_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state() if DEVICE.type == "cuda" else None,
            "train_sampler_rng_state": train_sampler.rng.get_state(),
            "train_sampler_epoch_order": train_sampler.epoch_order,
            "train_sampler_pos": train_sampler.pos,
        }
        ckpt_path = output_dir / "training_state_latest.pt"
        tmp_ckpt_path = ckpt_path.with_suffix(".pt.tmp")
        torch.save(ckpt, tmp_ckpt_path)
        os.replace(tmp_ckpt_path, ckpt_path)
        print(f"  -> Saved resumable training state to {ckpt_path}")

        # Export official architecture layout
        if CHECKPOINT_POLICY == "final_epoch" or (
            CHECKPOINT_POLICY == "best_validation" and epochs_without_improvement == 0
        ):
            model_out_path = output_dir / MODEL_FILENAME
            tmp_model_path = model_out_path.with_suffix(".pt.tmp")
            export_sd = (
                model.export_official_state_dict()
                if isinstance(model, PackedPopulationModel)
                else {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
            torch.save(export_sd, tmp_model_path)
            os.replace(tmp_model_path, model_out_path)
            print(f"  -> Saved official-layout checkpoint to {model_out_path}")

    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train Population State-Space Model")
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR), help="Path to OHSUSimPopulation")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR), help="Path to cache directory")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for model")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--benchmark-only", action="store_true", help="Run benchmark mode")
    parser.add_argument("--prepare-cache-only", action="store_true", help="Prepare cache and scalers, then exit")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--train-sequence-limit", type=int, default=TRAIN_SEQUENCE_LIMIT, help="Max unique sequences")
    parser.add_argument("--use-official-model", action="store_true", help="Disable packed model optimization")
    parser.add_argument("--run-validation", action="store_true", help="Run validation after each epoch")
    parser.add_argument("--use-reference-scalers", action="store_true", help="Use official reference scalers")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild cache")
    args = parser.parse_args()

    train(
        dataset_dir=Path(args.dataset_dir),
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        resume_checkpoint=args.resume or os.environ.get("RESUME_CHECKPOINT"),
        benchmark_only=args.benchmark_only or (os.environ.get("BENCHMARK_ONLY", "0") == "1"),
        prepare_cache_only=args.prepare_cache_only or (os.environ.get("PREPARE_CACHE_ONLY", "0") == "1"),
        use_packed_model=not args.use_official_model,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_sequence_limit=args.train_sequence_limit,
        run_validation=args.run_validation or (os.environ.get("RUN_VALIDATION_DURING_FINAL_TRAINING", "0") == "1"),
        use_reference_scalers=args.use_reference_scalers or (os.environ.get("USE_REFERENCE_SCALERS", "0") == "1"),
        force_cache_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
