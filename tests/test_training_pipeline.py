"""
tests/test_training_pipeline.py

Unit tests for sequence generation, window framing, trajectory boundary safety,
loss functions, and checkpoint saving/resumption.
"""

import os
import sys
import copy
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_population_model import (
    FramedWindowSampler,
    ForwardEulerSimulatorPop,
    PopulationLoss,
    STATE_WEIGHTS,
    C1_INDEX,
    SEQ_LEN,
    TRAIN_OVERLAP,
    BATCH_SIZE,
)
from t1dsim_ai.population_model import CGMOHSUSimStateSpaceModel_V2
from t1dsim_ai.options import n_neurons_pop


class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.T = 2016
        self.n_scenarios = 4
        self.n_states = 10
        self.n_inputs = 2

        # Create mock 3D scenario arrays (N, T, n_feat)
        np.random.seed(42)
        self.x_data = np.random.randn(self.n_scenarios, self.T, self.n_states).astype(np.float32)
        self.u_data = np.random.randn(self.n_scenarios, self.T, self.n_inputs).astype(np.float32)

    def test_sampler_framing_and_boundary_safety(self):
        seq_len = 61
        overlap = 0.75
        expected_hop = max(1, int(round((1 - overlap) * (seq_len - 1))))  # 15
        expected_starts_per_scenario = len(np.arange(0, self.T - seq_len + 1, expected_hop))  # 131
        total_expected_pairs = self.n_scenarios * expected_starts_per_scenario  # 4 * 131 = 524

        sampler = FramedWindowSampler(
            self.x_data, self.u_data, seq_len=seq_len, overlap=overlap,
            batch_size=16, device=self.device, seed=123
        )

        self.assertEqual(sampler.hop, expected_hop)
        self.assertEqual(len(sampler.pairs), total_expected_pairs)

        # Verify NO sequence start index exceeds T - seq_len (strict boundary protection)
        max_start = sampler.pairs[:, 1].max()
        self.assertLessEqual(max_start, self.T - seq_len)

        # Verify all scenario indices are within range [0, n_scenarios - 1]
        self.assertTrue((sampler.pairs[:, 0] >= 0).all())
        self.assertTrue((sampler.pairs[:, 0] < self.n_scenarios).all())

    def test_sampler_deterministic_reshuffle_and_cap(self):
        seq_limit = 200
        sampler = FramedWindowSampler(
            self.x_data, self.u_data, seq_len=61, overlap=0.75,
            batch_size=16, device=self.device, seed=999, max_sequences=seq_limit
        )
        self.assertEqual(len(sampler.pairs), seq_limit)

        # Verify reshuffle produces consistent sequence when reseeded
        order1 = sampler.epoch_order.copy()
        sampler._reshuffle()
        order2 = sampler.epoch_order.copy()
        # Different orders between epochs
        self.assertFalse(np.array_equal(order1, order2))

    def test_batch_from_pairs_shapes(self):
        sampler = FramedWindowSampler(
            self.x_data, self.u_data, seq_len=61, overlap=0.75,
            batch_size=16, device=self.device, seed=42
        )
        x0, u_batch, x_true = sampler.next_batch()

        # x0: (B, n_states) = (16, 10)
        self.assertEqual(x0.shape, (16, 10))
        # u_batch: (seq_len, B, n_inputs) = (61, 16, 2)
        self.assertEqual(u_batch.shape, (61, 16, 2))
        # x_true: (seq_len, B, n_states) = (61, 16, 10)
        self.assertEqual(x_true.shape, (61, 16, 10))

    def test_population_loss_mathematical_behavior(self):
        loss_fn = PopulationLoss(
            state_weights=STATE_WEIGHTS,
            state_min=np.zeros(10, dtype=np.float32),
            c1_index=C1_INDEX,
            alpha=0.7,
            beta=0.08
        ).to(self.device)

        # Synthetic simulation and target
        x_true = torch.ones(61, 8, 10)
        x_sim = torch.ones(61, 8, 10)

        # Zero error -> zero fit loss, zero consistency MSE, zero soft penalty
        l_total, l_fit, l_cons = loss_fn(x_sim, x_true, lim_inferior_scaled=0.0, lim_superior_scaled=2.0)
        self.assertAlmostEqual(l_total.item(), 0.0, places=5)

        # Under-prediction in hypo zone (y_true <= lim_inferior, y_sim > y_true) gets 2.0x penalty
        # Over-prediction in hyper zone (y_true >= lim_superior, y_sim < y_true) gets 6.0x penalty
        y_sim_hyper = torch.tensor([[[1.5]]])  # below 3.0
        y_true_hyper = torch.tensor([[[3.0]]]) # >= 2.0
        penalty_hyper = loss_fn.fit_penalty(y_sim_hyper, y_true_hyper, lim_inferior=0.0, lim_superior=2.0)
        self.assertEqual(penalty_hyper.item(), 6.0)

    def test_checkpoint_saving_and_resumption(self):
        temp_dir = tempfile.mkdtemp(prefix="test_ckpt_")
        try:
            model = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

            ckpt_path = os.path.join(temp_dir, "training_state_latest.pt")
            saved_state = {
                "completed_epoch": 5,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": 0.1234,
                "epochs_without_improvement": 2,
                "numpy_rng_state": np.random.get_state(),
                "torch_cpu_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state": None,
                "train_sampler_rng_state": np.random.get_state(),
                "train_sampler_epoch_order": np.arange(100),
                "train_sampler_pos": 32,
            }
            torch.save(saved_state, ckpt_path)

            # Test loading
            loaded = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.assertEqual(loaded["completed_epoch"], 5)
            self.assertEqual(loaded["best_val_loss"], 0.1234)
            self.assertEqual(loaded["train_sampler_pos"], 32)

            # Verify model weights restore exactly
            clone_model = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
            clone_model.load_state_dict(loaded["model_state_dict"])
            for p1, p2 in zip(model.parameters(), clone_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
