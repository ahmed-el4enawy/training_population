import os
import time
import numpy as np
import torch
import torch.optim as optim

from t1dsim_ai.population_model import CGMOHSUSimStateSpaceModel_V2
from t1dsim_ai.options import n_neurons_pop
from train_population_model import (
    get_dataset_dims, get_splits, fill_split_buffer,
    transform_in_place, FramedWindowSampler, ForwardEulerSimulatorPop,
    PopulationLoss, MERGED_MAT_PATH, OUTPUT_DIR, SEED, SEQ_LEN,
    INPUT_COLUMN_PERM, STATE_WEIGHTS, C1_INDEX, ALPHA, BETA,
    LR, LR_DECAY_PER_EPOCH
)
import h5py
from sklearn.preprocessing import RobustScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dummy_dataset():
    path = "dummy_dataset.mat"
    N, T, n_states, n_inputs = 10, 300, 10, 2
    states = np.random.rand(n_states, T, N).astype(np.float32)
    glucose = states[0, :, :] * 18.0 / 0.16
    with h5py.File(path, "w") as f:
        f.create_dataset("dataset_states", data=states)
        f.create_dataset("dataset_inputs", data=np.random.rand(n_inputs, T, N).astype(np.float32))
        f.create_dataset("dataset_glucose", data=glucose)
        f.create_dataset("dataset_split_id", data=np.zeros(N, dtype=np.int32))
    return path

def run_smoke_test():
    print("Starting smoke test...")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    mock_path = create_dummy_dataset()
    
    n_scenarios, T, n_states, n_inputs = get_dataset_dims(mock_path)
    train_idx, _, _ = get_splits(mock_path, max_train=5) # tiny subset
    
    x_train_flat, u_train_flat = fill_split_buffer(mock_path, train_idx, T, n_states, n_inputs)
    if INPUT_COLUMN_PERM is not None:
        u_train_flat = np.ascontiguousarray(u_train_flat[:, INPUT_COLUMN_PERM])
        
    scaler_states = RobustScaler().fit(x_train_flat)
    scaler_inputs = RobustScaler().fit(u_train_flat)
    
    transform_in_place(scaler_states, x_train_flat)
    transform_in_place(scaler_inputs, u_train_flat)
    
    x_train = x_train_flat.reshape(len(train_idx), T, n_states)
    u_train = u_train_flat.reshape(len(train_idx), T, n_inputs)
    
    state_min = x_train.reshape(-1, n_states).min(axis=0)
    state_min[C1_INDEX] = (0.0 - scaler_states.center_[C1_INDEX]) / scaler_states.scale_[C1_INDEX]
    
    # A. Model initialization
    try:
        model = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop).to(DEVICE)
        print("Model initialization: PASS")
    except Exception as e:
        print(f"Model initialization: FAIL - {e}")
        return
        
    simulator = ForwardEulerSimulatorPop(model, ts=5.0).to(DEVICE)
    loss_fn = PopulationLoss(STATE_WEIGHTS, state_min, C1_INDEX, alpha=ALPHA, beta=BETA).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY_PER_EPOCH)
    
    sampler = FramedWindowSampler(x_train, u_train, SEQ_LEN, 0.75, 4, DEVICE, seed=SEED)
    
    # B. Batch dimensions
    x0, u_batch, x_true = sampler.next_batch()
    try:
        assert x0.shape == (4, 10), f"x0 shape is {x0.shape}"
        assert u_batch.shape == (61, 4, 2), f"u_batch shape is {u_batch.shape}"
        assert x_true.shape == (61, 4, 10), f"x_true shape is {x_true.shape}"
        print("Batch shapes: PASS")
    except AssertionError as e:
        print(f"Batch shapes: FAIL - {e}")
        
    # C. Euler rollout
    x_sim = simulator(x0, u_batch)
    try:
        assert x_sim.shape == (61, 4, 10), f"x_sim shape is {x_sim.shape}"
        print("Forward Euler: PASS")
    except AssertionError as e:
        print(f"Forward Euler: FAIL - {e}")
        
    # D. Finite forward pass
    try:
        assert torch.isfinite(x_sim).all(), "Simulation output has non-finite values"
        print("Finite simulation: PASS")
    except AssertionError as e:
        print(f"Finite simulation: FAIL - {e}")
        
    # E. Finite loss
    loss, l_fit, l_cons = loss_fn(x_sim, x_true, -10.0, 10.0) # mock limits
    try:
        assert torch.isfinite(loss), "Loss is non-finite"
        print("Finite loss: PASS")
    except AssertionError as e:
        print(f"Finite loss: FAIL - {e}")
        
    # F. Backward pass
    try:
        loss.backward()
        print("Backward: PASS")
    except Exception as e:
        print(f"Backward: FAIL - {e}")
        
    # G. Finite gradients
    try:
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "Gradient has non-finite values"
        print("Finite gradients: PASS")
    except AssertionError as e:
        print(f"Finite gradients: FAIL - {e}")
        
    # H. Optimizer step
    try:
        optimizer.step()
        print("Optimizer step: PASS")
    except Exception as e:
        print(f"Optimizer step: FAIL - {e}")
        
    # I. C1 rule
    try:
        err = x_sim[1:] - x_true[1:]
        mse_per_state = torch.mean(err ** 2, dim=(0, 1))
        mask = torch.ones_like(mse_per_state)
        mask[C1_INDEX] = 0.0
        mse_masked = mse_per_state * mask
        assert mse_masked[C1_INDEX] == 0.0, "C1 MSE is not exactly zero"
        print("C1 MSE=0: PASS")
    except AssertionError as e:
        print(f"C1 MSE=0: FAIL - {e}")
        
    # J. LR scheduler
    try:
        scheduler.step()
        assert np.isclose(optimizer.param_groups[0]['lr'], 1e-3 * np.exp(-0.1)), "LR scheduler mismatch"
        print("LR scheduler: PASS")
    except AssertionError as e:
        print(f"LR scheduler: FAIL - {e}")
        
    # K. Timing probe reproducibility
    try:
        t_probe = time.time()
        probe_pos = sampler.pos
        probe_epoch_order = sampler.epoch_order.copy()
        probe_rng_state = sampler.rng.get_state()
        
        # mock probe execution
        with torch.no_grad():
            x0_p, u_batch_p, x_true_p = sampler.next_batch()
            x_sim_p = simulator(x0_p, u_batch_p)
            loss_fn(x_sim_p, x_true_p, -10.0, 10.0)
            
        sampler.rng.set_state(probe_rng_state)
        sampler.epoch_order = probe_epoch_order
        sampler.pos = probe_pos
        
        assert sampler.pos == probe_pos, "sampler.pos changed"
        np.testing.assert_array_equal(sampler.epoch_order, probe_epoch_order, err_msg="sampler.epoch_order changed")
        
        # if we reach here without exceptions, pass
        print("Timing probe state restoration: PASS")
    except AssertionError as e:
        print(f"Timing probe state restoration: FAIL - {e}")
        
    print("Smoke test completed.")

if __name__ == "__main__":
    run_smoke_test()
