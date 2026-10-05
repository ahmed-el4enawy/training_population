import os
import time
import numpy as np
import h5py
from sklearn.preprocessing import RobustScaler
import torch

from t1dsim_ai.utils.preprocess import scale_single_state, scale_inverse_Q1
from train_population_model import (
    MERGED_MAT_PATH, get_dataset_dims, get_splits, _check_q1_channel, _MGDL_PER_MMOL, _VDG,
    FramedWindowSampler, STATE_WEIGHTS, C1_INDEX, ALPHA, BETA, DEVICE, CHUNK_SCENARIOS
)
from sklearn.preprocessing._data import _handle_zeros_in_scale

CACHE_DIR = r"E:\population_training_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ----------------- OLD IN-MEMORY LOGIC -----------------
def old_fill_split_buffer(path, idx, T, n_states, n_inputs, chunk_size=20):
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
                
                implied = st[..., 0] * _MGDL_PER_MMOL / _VDG
                err = float(np.abs(implied - gl).max())
                if err > 1.0: raise RuntimeError(f"Q1 mismatch {err}")
                
                st[..., 0] = gl
                x_flat[pos * T:(pos + c) * T] = st.reshape(c * T, n_states)
                u_flat[pos * T:(pos + c) * T] = ip.reshape(c * T, n_inputs)
                pos += c
    return x_flat, u_flat

# ----------------- NEW MEMMAP LOGIC -----------------
def new_fill_split_buffer_memmap(path, idx, T, n_states, n_inputs, prefix, chunk_size=1000):
    idx = np.asarray(idx)
    total_rows = len(idx) * T
    
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
                
                implied = st[..., 0] * _MGDL_PER_MMOL / _VDG
                err = float(np.abs(implied - gl).max())
                if err > 1.0: raise RuntimeError(f"Q1 mismatch {err}")
                
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
        # Fit one feature at a time to save memory
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

def transform_in_place_memmap(scaler, memmap_array, prefix, name, chunk_rows=500_000):
    n = memmap_array.shape[0]
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        # scaler.transform produces float64 by default, cast back to float32
        memmap_array[start:end] = scaler.transform(memmap_array[start:end]).astype(np.float32)
        pct = int(end / n * 100)
        if (start // chunk_rows) % max(1, (n // chunk_rows // 20)) == 0:
            print(f"Scaling {prefix} {name}: {pct}%", flush=True)
    memmap_array.flush()
    print(f"Scaling {prefix} {name}: 100%", flush=True)

def compute_state_min_memmap(memmap_array, n_states, chunk_rows=500_000):
    n = memmap_array.shape[0]
    global_min = np.full(n_states, np.inf, dtype=np.float32)
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        chunk_min = memmap_array[start:end].min(axis=0)
        global_min = np.minimum(global_min, chunk_min)
    return global_min

def test_equivalence():
    print("Extracting small subset for equivalence check...")
    # Use just 200 scenarios
    n_scenarios, T, n_states, n_inputs = get_dataset_dims(MERGED_MAT_PATH)
    train_idx, _, _ = get_splits(MERGED_MAT_PATH, max_train=200)
    
    # OLD
    x_old, u_old = old_fill_split_buffer(MERGED_MAT_PATH, train_idx, T, n_states, n_inputs)
    x_old_raw = x_old.copy()
    u_old_raw = u_old.copy()
    
    scaler_states_old = RobustScaler().fit(x_old)
    scaler_inputs_old = RobustScaler().fit(u_old)
    
    x_old[:] = scaler_states_old.transform(x_old).astype(np.float32)
    u_old[:] = scaler_inputs_old.transform(u_old).astype(np.float32)
    
    state_min_old = x_old.reshape(-1, n_states).min(axis=0)
    state_min_old[C1_INDEX] = (0.0 - scaler_states_old.center_[C1_INDEX]) / scaler_states_old.scale_[C1_INDEX]
    
    # NEW
    x_new, u_new = new_fill_split_buffer_memmap(MERGED_MAT_PATH, train_idx, T, n_states, n_inputs, prefix="TEST_TRAIN", chunk_size=20)
    
    # Raw tests
    err_x = np.abs(x_old_raw - x_new).max()
    err_u = np.abs(u_old_raw - u_new).max()
    print(f"raw states diff: {err_x}")
    print(f"raw inputs diff: {err_u}")
    assert err_x == 0
    assert err_u == 0
    
    scaler_states_new = fit_robust_scaler_memmap(x_new)
    scaler_inputs_new = fit_robust_scaler_memmap(u_new)
    
    print(f"scaler center diff: {np.abs(scaler_states_old.center_ - scaler_states_new.center_).max()}")
    print(f"scaler scale diff: {np.abs(scaler_states_old.scale_ - scaler_states_new.scale_).max()}")
    assert np.allclose(scaler_states_old.center_, scaler_states_new.center_)
    assert np.allclose(scaler_states_old.scale_, scaler_states_new.scale_)
    
    transform_in_place_memmap(scaler_states_new, x_new, "TEST_TRAIN", "states")
    transform_in_place_memmap(scaler_inputs_new, u_new, "TEST_TRAIN", "inputs")
    
    err_sx = np.abs(x_old - x_new).max()
    err_su = np.abs(u_old - u_new).max()
    print(f"scaled states diff: {err_sx}")
    print(f"scaled inputs diff: {err_su}")
    assert err_sx < 1e-5
    assert err_su < 1e-5
    
    state_min_new = compute_state_min_memmap(x_new, n_states)
    state_min_new[C1_INDEX] = (0.0 - scaler_states_new.center_[C1_INDEX]) / scaler_states_new.scale_[C1_INDEX]
    print(f"state_min diff: {np.abs(state_min_old - state_min_new).max()}")
    assert np.allclose(state_min_old, state_min_new, atol=1e-5)
    
    # Check reshape shares memory
    x_new_reshaped = x_new.reshape(len(train_idx), T, n_states)
    u_new_reshaped = u_new.reshape(len(train_idx), T, n_inputs)
    assert np.shares_memory(x_new, x_new_reshaped)
    assert np.shares_memory(u_new, u_new_reshaped)
    
    # Sampler tests
    x_old_reshaped = x_old.reshape(len(train_idx), T, n_states)
    u_old_reshaped = u_old.reshape(len(train_idx), T, n_inputs)
    
    sampler_old = FramedWindowSampler(x_old_reshaped, u_old_reshaped, seq_len=61, overlap=0.75, batch_size=128, device=DEVICE, seed=0)
    sampler_new = FramedWindowSampler(x_new_reshaped, u_new_reshaped, seq_len=61, overlap=0.75, batch_size=128, device=DEVICE, seed=0)
    
    assert np.array_equal(sampler_old.pairs, sampler_new.pairs)
    
    x0_o, ub_o, xt_o = sampler_old.next_batch()
    x0_n, ub_n, xt_n = sampler_new.next_batch()
    
    assert torch.allclose(x0_o, x0_n, atol=1e-5)
    assert torch.allclose(ub_o, ub_n, atol=1e-5)
    assert torch.allclose(xt_o, xt_n, atol=1e-5)
    
    print("EQUIVALENCE CHECK PASS")

if __name__ == "__main__":
    test_equivalence()
