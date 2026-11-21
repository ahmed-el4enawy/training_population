"""
tests/test_dataset_pipeline.py

Unit tests for dataset discovery, column-to-state mapping, split isolation,
memory-mapped cache creation, validation, and error handling.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root and T1DSim_AI are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_pipeline import (
    STATE_ORDER,
    INPUT_ORDER,
    RAW_STATE_COLUMNS,
    TRAJECTORY_LENGTH,
    TOTAL_PROBLEMS_PER_FILE,
    EXPECTED_ROWS_PER_FILE,
    discover_population_dataset,
    parse_single_pkl_file,
    build_memmap_cache,
    validate_cache,
    load_cached_split,
)


def create_mock_dataframe(with_correction_bolus=False, num_problems=11):
    rows = num_problems * TRAJECTORY_LENGTH
    data = {
        "time": np.arange(rows, dtype=np.int32),
        "dosing_problem_id": np.repeat(np.arange(num_problems), TRAJECTORY_LENGTH),
        "sim_day": np.repeat(np.tile(np.arange(1, 8), 288), num_problems)[:rows],
        "initCGM": np.full(rows, 140.0, dtype=np.float32),
        "sim_hour_of_day": np.zeros(rows, dtype=np.int32),
        "input_insulin": np.random.uniform(0.5, 2.0, rows).astype(np.float32),
        "input_meal_carbs": np.random.uniform(0.0, 50.0, rows).astype(np.float32),
        "output_glucose": np.random.uniform(70.0, 180.0, rows).astype(np.float32),
        "output_cgm": np.random.uniform(70.0, 180.0, rows).astype(np.float32),
        "state_Q1": np.random.uniform(1.0, 3.0, rows).astype(np.float32),
        "state_Q2": np.random.uniform(0.5, 1.5, rows).astype(np.float32),
        "state_S1": np.random.uniform(10.0, 15.0, rows).astype(np.float32),
        "state_S2": np.random.uniform(10.0, 15.0, rows).astype(np.float32),
        "state_I": np.random.uniform(10.0, 20.0, rows).astype(np.float32),
        "state_X1": np.random.uniform(0.01, 0.05, rows).astype(np.float32),
        "state_X2": np.random.uniform(0.001, 0.01, rows).astype(np.float32),
        "state_X3": np.random.uniform(0.1, 0.5, rows).astype(np.float32),
        "state_Ug": np.random.uniform(0.0, 1.0, rows).astype(np.float32),
    }
    if with_correction_bolus:
        # Give only the first 2 dosing problems some correction boluses
        cb = np.zeros(rows, dtype=np.float32)
        cb[: 2 * TRAJECTORY_LENGTH] = np.random.uniform(0.0, 1.0, 2 * TRAJECTORY_LENGTH)
        data["correction_bolus"] = cb

    return pd.DataFrame(data)


class TestDatasetPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_dataset_pipeline_")
        self.ds_dir = Path(self.temp_dir) / "OHSUSimPopulation"
        self.cache_dir = Path(self.temp_dir) / "cache"

        # Create realistic directory structure
        self.sim002_train = self.ds_dir / "SIM002" / "POPULATION" / "TRAIN"
        self.sim002_val = self.ds_dir / "SIM002" / "POPULATION" / "VALIDATION"
        self.sim002_test = self.ds_dir / "SIM002" / "POPULATION" / "TEST"
        self.sim004_train = self.ds_dir / "SIM004" / "POPULATION" / "boluslate_5min" / "TRAIN"
        self.sim005_train = self.ds_dir / "SIM005" / "POPULATION" / "bolusCorrectionFactor_28" / "TRAIN"

        for p in [self.sim002_train, self.sim002_val, self.sim002_test, self.sim004_train, self.sim005_train]:
            p.mkdir(parents=True, exist_ok=True)

        # Write mock files
        df_standard = create_mock_dataframe(with_correction_bolus=False)
        df_standard.to_pickle(self.sim002_train / "S_001003mealScenario_0_initCGM_0_140.pkl")
        df_standard.to_pickle(self.sim002_val / "S_001004mealScenario_0_initCGM_0_140.pkl")
        df_standard.to_pickle(self.sim002_test / "S_042006mealScenario_0_initCGM_0_140.pkl")
        df_standard.to_pickle(self.sim004_train / "S_001003mealScenario_0_initCGM_0_140.pkl")

        df_cb = create_mock_dataframe(with_correction_bolus=True)
        df_cb.to_pickle(self.sim005_train / "001003mealScenario_0_initCGM_0_140.pkl")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_discovery_and_split_isolation(self):
        discovery = discover_population_dataset(self.ds_dir)
        self.assertEqual(len(discovery["sim002_train"]), 1)
        self.assertEqual(len(discovery["sim002_val"]), 1)
        self.assertEqual(len(discovery["sim002_test"]), 1)
        self.assertEqual(len(discovery["sim004_train"]), 1)
        self.assertEqual(len(discovery["sim005_train"]), 1)
        self.assertEqual(discovery["total_files"], 5)

    def test_leakage_detection_raises_error(self):
        # Place a test subject file in SIM002 TRAIN
        leaked_file = self.sim002_train / "S_042006mealScenario_1_initCGM_0_140.pkl"
        create_mock_dataframe().to_pickle(leaked_file)

        with self.assertRaises(ValueError) as ctx:
            discover_population_dataset(self.ds_dir)
        self.assertIn("Subject leakage detected", str(ctx.exception))

    def test_parse_single_pkl_file_mapping(self):
        f = self.sim002_train / "S_001003mealScenario_0_initCGM_0_140.pkl"
        x, u, n = parse_single_pkl_file(f, is_sim005=False)

        self.assertEqual(n, 11)
        self.assertEqual(x.shape, (11, 2016, 10))
        self.assertEqual(u.shape, (11, 2016, 2))

        # Check 10th state C1 (state_Ug2) is identically zero
        self.assertTrue((x[:, :, 9] == 0.0).all())
        # Check 1st state is output_cgm
        df = pd.read_pickle(f)
        cgm_first_problem = df[df["dosing_problem_id"] == 0]["output_cgm"].values.astype(np.float32)
        np.testing.assert_allclose(x[0, :, 0], cgm_first_problem)

    def test_sim005_correction_bolus_filtering(self):
        f = self.sim005_train / "001003mealScenario_0_initCGM_0_140.pkl"
        x, u, n = parse_single_pkl_file(f, is_sim005=True)

        # In create_mock_dataframe, only first 2 dosing problems had correction_bolus > 0
        self.assertEqual(n, 2)
        self.assertEqual(x.shape, (2, 2016, 10))
        self.assertEqual(u.shape, (2, 2016, 2))

    def test_cache_build_validate_and_load(self):
        manifest = build_memmap_cache(self.ds_dir, self.cache_dir, force_rebuild=True)
        self.assertTrue((self.cache_dir / "manifest.json").exists())
        self.assertTrue((self.cache_dir / "TRAIN_states.dat").exists())
        self.assertTrue((self.cache_dir / "TRAIN_inputs.dat").exists())

        valid, m_loaded = validate_cache(self.cache_dir)
        self.assertTrue(valid)
        self.assertEqual(m_loaded["CACHE_SCHEMA_VERSION"], 3)

        # Load split
        x_mmap, u_mmap, _ = load_cached_split(self.cache_dir, "TRAIN")
        expected_train_traj = 11 + 11 + 2  # SIM002(11) + SIM004(11) + SIM005(2)
        self.assertEqual(x_mmap.shape, (expected_train_traj * 2016, 10))
        self.assertEqual(u_mmap.shape, (expected_train_traj * 2016, 2))

    def test_corrupted_file_error_handling(self):
        bad_file = self.sim002_train / "S_001005mealScenario_0_initCGM_0_140.pkl"
        with open(bad_file, "w") as fh:
            fh.write("corrupt content")

        with self.assertRaises(Exception):
            parse_single_pkl_file(bad_file)


if __name__ == "__main__":
    unittest.main()
