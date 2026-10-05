import os
import h5py
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing._data import _handle_zeros_in_scale

# Mock constants for test
SEQ_LEN = 61
TRAIN_OVERLAP = 0.75
BATCH_SIZE = 128
C1_INDEX = 1
Q1_INDEX = 0

def old_in_memory_path(h5_path, n_traces=1000):
    with h5py.File(h5_path, 'r') as f:
        ds_s = f['dataset_states']
        ds_i = f['dataset_inputs']
        ds_g = f['dataset_glucose']
        
        st = np.transpose(ds_s[:, :, 0:n_traces], (2, 1, 0)).astype(np.float32, order='C')
        ip = np.transpose(ds_i[:, :, 0:n_traces], (2, 1, 0)).astype(np.float32, order='C')
        gl = np.transpose(ds_g[:, 0:n_traces], (1, 0)).astype(np.float32, order='C')
        
    st[..., 0] = gl
    
    x_train_flat = st.reshape(-1, 10).copy()
    u_train_flat = ip.reshape(-1, 2).copy()
    
    scaler_states = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    scaler_inputs = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    
    scaler_states.fit(x_train_flat)
    scaler_inputs.fit(u_train_flat)
    
    x_train_flat = scaler_states.transform(x_train_flat)
    u_train_flat = scaler_inputs.transform(u_train_flat)
    
    state_min = x_train_flat.min(axis=0)
    state_min[C1_INDEX] = (0.0 - scaler_states.center_[C1_INDEX]) / scaler_states.scale_[C1_INDEX]
    
    x_train = x_train_flat.reshape(n_traces, 2016, 10)
    u_train = u_train_flat.reshape(n_traces, 2016, 2)
    return st, ip, scaler_states, scaler_inputs, x_train, u_train, state_min

def new_memmap_path(h5_path, cache_dir, n_traces=1000):
    os.makedirs(cache_dir, exist_ok=True)
    st_path = os.path.join(cache_dir, 'st.dat')
    ip_path = os.path.join(cache_dir, 'ip.dat')
    
    with h5py.File(h5_path, 'r') as f:
        ds_s = f['dataset_states']
        ds_i = f['dataset_inputs']
        ds_g = f['dataset_glucose']
        
        st_mmap = np.memmap(st_path, dtype=np.float32, mode='w+', shape=(n_traces, 2016, 10))
        ip_mmap = np.memmap(ip_path, dtype=np.float32, mode='w+', shape=(n_traces, 2016, 2))
        
        # Load exactly as old path
        st_tmp = np.transpose(ds_s[:, :, 0:n_traces], (2, 1, 0)).astype(np.float32, order='C')
        ip_tmp = np.transpose(ds_i[:, :, 0:n_traces], (2, 1, 0)).astype(np.float32, order='C')
        gl_tmp = np.transpose(ds_g[:, 0:n_traces], (1, 0)).astype(np.float32, order='C')
        
        st_tmp[..., 0] = gl_tmp
        st_mmap[:] = st_tmp
        ip_mmap[:] = ip_tmp
        st_mmap.flush()
        ip_mmap.flush()
        
    x_train_flat = np.memmap(st_path, dtype=np.float32, mode='r+', shape=(n_traces*2016, 10))
    u_train_flat = np.memmap(ip_path, dtype=np.float32, mode='r+', shape=(n_traces*2016, 2))
    
    # Custom robust scaler
    def fit_robust_scaler_memmap(array):
        n_rows, n_features = array.shape
        centers = np.zeros(n_features, dtype=np.float64)
        scales = np.zeros(n_features, dtype=np.float64)
        for i in range(n_features):
            q25, median, q75 = np.percentile(array[:, i], [25.0, 50.0, 75.0])
            centers[i] = median
            scales[i] = q75 - q25
        scales = _handle_zeros_in_scale(scales, copy=False)
        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
        scaler.center_ = centers
        scaler.scale_ = scales
        scaler.n_features_in_ = n_features
        return scaler
        
    scaler_states = fit_robust_scaler_memmap(x_train_flat)
    scaler_inputs = fit_robust_scaler_memmap(u_train_flat)
    
    # Transform in place chunked
    def transform_in_place(scaler, array):
        n = array.shape[0]
        chunk = 500_000
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            array[start:end] = scaler.transform(array[start:end])
        array.flush()
        
    transform_in_place(scaler_states, x_train_flat)
    transform_in_place(scaler_inputs, u_train_flat)
    
    def compute_state_min_memmap(array, n_states):
        n = array.shape[0]
        global_min = np.full(n_states, np.inf, dtype=np.float32)
        chunk = 500_000
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            global_min = np.minimum(global_min, array[start:end].min(axis=0))
        return global_min
        
    state_min = compute_state_min_memmap(x_train_flat, 10)
    state_min[C1_INDEX] = (0.0 - scaler_states.center_[C1_INDEX]) / scaler_states.scale_[C1_INDEX]
    
    x_train = np.memmap(st_path, dtype=np.float32, mode='r', shape=(n_traces, 2016, 10))
    u_train = np.memmap(ip_path, dtype=np.float32, mode='r', shape=(n_traces, 2016, 2))
    
    return st_tmp, ip_tmp, scaler_states, scaler_inputs, x_train, u_train, state_min

if __name__ == "__main__":
    h5_path = r"E:\T1D_population_training\population_development_dataset_merged.mat"
    cache_dir = r"E:\T1D_population_training\test_cache"
    
    
    
    
    
    st_old, ip_old, ss_old, si_old, x_old, u_old, min_old = old_in_memory_path(h5_path)
    
    
    st_new, ip_new, ss_new, si_new, x_new, u_new, min_new = new_memmap_path(h5_path, cache_dir)
    
    
    print("\n--- EQUIVALENCE RESULTS ---")
    print("Raw states max abs diff:", np.max(np.abs(st_old - st_new)))
    print("Raw inputs max abs diff:", np.max(np.abs(ip_old - ip_new)))
    print("Q1-replaced state channel max abs diff:", np.max(np.abs(st_old[...,0] - st_new[...,0])))
    
    print("Scaler states center equal:", np.allclose(ss_old.center_, ss_new.center_))
    print("Scaler states scale equal:", np.allclose(ss_old.scale_, ss_new.scale_))
    print("Scaler inputs center equal:", np.allclose(si_old.center_, si_new.center_))
    print("Scaler inputs scale equal:", np.allclose(si_old.scale_, si_new.scale_))
    
    print("Scaled states max abs diff:", np.max(np.abs(x_old - x_new)))
    print("Scaled inputs max abs diff:", np.max(np.abs(u_old - u_new)))
    print("state_min max abs diff:", np.max(np.abs(min_old - min_new)))
    
    # Test FramedWindowSampler
    import sys
    sys.path.append(r"F:\College\Graduation Project\Sheno\training_population")
    from T1DSim_AI.data.population_sampler import FramedWindowSampler
    
    np.random.seed(42)
    torch.manual_seed(42)
    sampler_old = FramedWindowSampler(x_old, u_old, SEQ_LEN, TRAIN_OVERLAP, BATCH_SIZE, shuffle=True)
    batch_old = next(iter(sampler_old))
    
    np.random.seed(42)
    torch.manual_seed(42)
    sampler_new = FramedWindowSampler(x_new, u_new, SEQ_LEN, TRAIN_OVERLAP, BATCH_SIZE, shuffle=True)
    batch_new = next(iter(sampler_new))
    
    # sampler_old.window_pairs vs sampler_new.window_pairs
    pairs_old = sampler_old.window_pairs
    pairs_new = sampler_new.window_pairs
    print("Window pairs identical:", np.array_equal(pairs_old, pairs_new))
    
    print("First selected x0 identical:", np.allclose(batch_old[0], batch_new[0]))
    print("First selected u_batch identical:", np.allclose(batch_old[1], batch_new[1]))
    print("First selected x_true identical:", np.allclose(batch_old[2], batch_new[2]))
