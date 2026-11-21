"""
dataset_pipeline.py

Robust, scientifically verified dataset ingestion and memory-mapped cache pipeline
for the reproduced OHSU glucoregulatory population dataset (SIM002, SIM004, SIM005).

Handles:
1. Discovery of all 5,010 .pkl files across SIM002, SIM004, and SIM005.
2. Strict isolation of SIM002 TEST (5 subjects) and VALIDATION (5 subjects) from TRAIN.
3. Accurate 10-state extraction:
   [Q1 (output_cgm), Q2, S1, S2, I, X1, X2, X3, C2 (state_Ug), C1 (state_Ug2 = 0.0)]
4. Accurate 2-input extraction:
   [input_insulin (U/h), input_meal_carbs (g)]
5. Trajectory preservation (each dosing problem is exactly 2,016 samples = 7 days at ts=5 min).
6. Filtering of SIM005 correction bolus sequences (retaining only trajectories with correction_bolus > 0).
7. Streaming generation of memory-mapped NumPy binary caches (TRAIN, VAL, TEST) with zero RAM bloat.
8. Cache manifest tracking with schema versioning, file counts, and metadata.
"""

import os
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
CACHE_SCHEMA_VERSION = 3
TRAJECTORY_LENGTH = 2016  # 7 days * 288 samples/day at ts=5 min
TOTAL_PROBLEMS_PER_FILE = 11
EXPECTED_ROWS_PER_FILE = TOTAL_PROBLEMS_PER_FILE * TRAJECTORY_LENGTH  # 22,176

STATE_ORDER = ["Q1", "Q2", "S1", "S2", "I", "X1", "X2", "X3", "C2", "C1"]
INPUT_ORDER = ["input_insulin", "input_meal_carbs"]

RAW_STATE_COLUMNS = [
    "output_cgm",
    "state_Q2",
    "state_S1",
    "state_S2",
    "state_I",
    "state_X1",
    "state_X2",
    "state_X3",
    "state_Ug",
]

DEFAULT_DATASET_DIR = Path(
    os.environ.get(
        "DATASET_DIR",
        "F:/College/Graduation Project/CodeForGeneratingDataset/GeneratedDataset/OHSUSimPopulation",
    )
)
DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "CACHE_DIR",
        "F:/College/Graduation Project/training_population/cache",
    )
)

# Known held-out validation and test subjects per paper Table 1
HELD_OUT_SUBJECTS_TEST = {"042006", "042007", "042011", "042012", "042013"}
HELD_OUT_SUBJECTS_VAL = {"001004", "001007", "001013", "001017", "001019"}


def extract_subject_id_from_filename(filename: str) -> str:
    """Extract 6-digit subject ID string from standard filename."""
    base = Path(filename).name
    if "mealScenario" in base:
        sub_part = base.split("mealScenario")[0]
        sub_id = sub_part.replace("S_", "")
        return sub_id
    return ""


def discover_population_dataset(dataset_dir: Path) -> Dict[str, List[Path]]:
    """
    Discover and categorize all .pkl files in the dataset directory.
    Validates subject isolation and returns sorted file lists per split.
    """
    dataset_dir = Path(dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {dataset_dir}")

    # SIM002 splits
    sim002_train_dir = dataset_dir / "SIM002" / "POPULATION" / "TRAIN"
    sim002_val_dir = dataset_dir / "SIM002" / "POPULATION" / "VALIDATION"
    sim002_test_dir = dataset_dir / "SIM002" / "POPULATION" / "TEST"

    sim002_train = sorted(list(sim002_train_dir.glob("*.pkl"))) if sim002_train_dir.exists() else []
    sim002_val = sorted(list(sim002_val_dir.glob("*.pkl"))) if sim002_val_dir.exists() else []
    sim002_test = sorted(list(sim002_test_dir.glob("*.pkl"))) if sim002_test_dir.exists() else []

    # SIM004 (all boluslate directories are training data)
    sim004_dir = dataset_dir / "SIM004"
    sim004_train = sorted(list(sim004_dir.glob("**/*.pkl"))) if sim004_dir.exists() else []

    # SIM005 (all correction factor directories are training data)
    sim005_dir = dataset_dir / "SIM005"
    sim005_train = sorted(list(sim005_dir.glob("**/*.pkl"))) if sim005_dir.exists() else []

    # Strict isolation check: TEST subjects must never appear in TRAIN
    train_files = sim002_train + sim004_train + sim005_train
    val_subjects = {extract_subject_id_from_filename(f.name) for f in sim002_val}
    test_subjects = {extract_subject_id_from_filename(f.name) for f in sim002_test}

    # Verify no accidental subject leakage
    for f in sim002_train:
        sid = extract_subject_id_from_filename(f.name)
        if sid in test_subjects:
            raise ValueError(f"Subject leakage detected! Test subject {sid} found in SIM002 TRAIN file: {f.name}")
        if sid in val_subjects:
            raise ValueError(f"Subject leakage detected! Val subject {sid} found in SIM002 TRAIN file: {f.name}")

    total_discovered = len(sim002_train) + len(sim002_val) + len(sim002_test) + len(sim004_train) + len(sim005_train)

    return {
        "sim002_train": sim002_train,
        "sim002_val": sim002_val,
        "sim002_test": sim002_test,
        "sim004_train": sim004_train,
        "sim005_train": sim005_train,
        "total_files": total_discovered,
    }


def parse_single_pkl_file(
    file_path: Path, is_sim005: bool = False
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Parse a single .pkl file into state matrix (N_p, 2016, 10) and input matrix (N_p, 2016, 2).
    For SIM005, only trajectories where correction_bolus > 0 are retained.
    """
    df = pd.read_pickle(file_path)

    if df.shape[0] != EXPECTED_ROWS_PER_FILE:
        raise ValueError(
            f"File {file_path.name} has invalid row count: {df.shape[0]} (expected {EXPECTED_ROWS_PER_FILE})"
        )

    # Fast block extraction
    x_raw = df[RAW_STATE_COLUMNS].values.astype(np.float32).reshape(TOTAL_PROBLEMS_PER_FILE, TRAJECTORY_LENGTH, 9)
    u_raw = df[INPUT_ORDER].values.astype(np.float32).reshape(TOTAL_PROBLEMS_PER_FILE, TRAJECTORY_LENGTH, 2)

    # Add 10th state: C1 (state_Ug2 = 0.0)
    x_10 = np.zeros((TOTAL_PROBLEMS_PER_FILE, TRAJECTORY_LENGTH, 10), dtype=np.float32)
    x_10[:, :, :9] = x_raw
    x_10[:, :, 9] = 0.0

    if is_sim005:
        if "correction_bolus" not in df.columns:
            raise ValueError(f"SIM005 file {file_path.name} is missing 'correction_bolus' column.")
        cb = df["correction_bolus"].values.reshape(TOTAL_PROBLEMS_PER_FILE, TRAJECTORY_LENGTH)
        has_cb = (cb > 0.0).any(axis=1)
        x_out = x_10[has_cb]
        u_out = u_raw[has_cb]
    else:
        x_out = x_10
        u_out = u_raw

    # Sanity checks
    if np.isnan(x_out).any() or np.isinf(x_out).any():
        raise ValueError(f"File {file_path.name} contains NaN or Inf in state variables.")
    if np.isnan(u_out).any() or np.isinf(u_out).any():
        raise ValueError(f"File {file_path.name} contains NaN or Inf in input variables.")

    return x_out, u_out, x_out.shape[0]


def count_dataset_trajectories(discovery: Dict[str, List[Path]], sample_sim005: bool = False) -> Dict[str, int]:
    """
    Count the exact number of trajectories across all splits.
    For SIM005, scans all files to get the exact count of active correction bolus trajectories.
    """
    val_count = len(discovery["sim002_val"]) * TOTAL_PROBLEMS_PER_FILE
    test_count = len(discovery["sim002_test"]) * TOTAL_PROBLEMS_PER_FILE
    sim002_train_count = len(discovery["sim002_train"]) * TOTAL_PROBLEMS_PER_FILE
    sim004_train_count = len(discovery["sim004_train"]) * TOTAL_PROBLEMS_PER_FILE

    print(f"Counting active trajectories in {len(discovery['sim005_train'])} SIM005 files...")
    sim005_active = 0
    t0 = time.time()
    for idx, f in enumerate(discovery["sim005_train"]):
        df = pd.read_pickle(f)
        cb = df["correction_bolus"].values.reshape(TOTAL_PROBLEMS_PER_FILE, TRAJECTORY_LENGTH)
        sim005_active += int((cb > 0.0).any(axis=1).sum())
        if (idx + 1) % 200 == 0 or (idx + 1) == len(discovery["sim005_train"]):
            print(f"  SIM005 scan: {idx + 1}/{len(discovery['sim005_train'])} files scanned [{time.time() - t0:.1f}s]")

    train_count = sim002_train_count + sim004_train_count + sim005_active

    return {
        "train_trajectories": train_count,
        "val_trajectories": val_count,
        "test_trajectories": test_count,
        "sim002_train_trajectories": sim002_train_count,
        "sim004_train_trajectories": sim004_train_count,
        "sim005_train_trajectories": sim005_active,
    }


def build_memmap_cache(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_rebuild: bool = False,
    include_sim004: bool = True,
    include_sim005: bool = True,
) -> Dict:
    """
    Stream source .pkl files into binary memory-mapped arrays on disk.
    Creates:
      - TRAIN_states.dat (N_train * 2016, 10)
      - TRAIN_inputs.dat (N_train * 2016, 2)
      - VAL_states.dat   (N_val * 2016, 10)
      - VAL_inputs.dat   (N_val * 2016, 2)
      - TEST_states.dat  (N_test * 2016, 10)
      - TEST_inputs.dat  (N_test * 2016, 2)
      - manifest.json
    """
    dataset_dir = Path(dataset_dir).resolve()
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cache_dir / "manifest.json"
    if not force_rebuild and manifest_path.exists():
        valid, manifest = validate_cache(cache_dir, expected_dataset_dir=dataset_dir)
        if valid:
            print(f"Valid cache already exists in {cache_dir}. Skipping rebuild.")
            return manifest
        else:
            print("Existing cache is invalid or stale. Rebuilding...")

    print("=" * 70)
    print("STARTING DATASET INGESTION & CACHE BUILD")
    print(f"Dataset root: {dataset_dir}")
    print(f"Cache target: {cache_dir}")
    print("=" * 70)

    discovery = discover_population_dataset(dataset_dir)
    print(f"Discovered {discovery['total_files']} total .pkl files:")
    print(f"  SIM002 TRAIN: {len(discovery['sim002_train'])} files")
    print(f"  SIM002 VAL:   {len(discovery['sim002_val'])} files")
    print(f"  SIM002 TEST:  {len(discovery['sim002_test'])} files")
    print(f"  SIM004 TRAIN: {len(discovery['sim004_train'])} files")
    print(f"  SIM005 TRAIN: {len(discovery['sim005_train'])} files")

    # Filter files according to options
    train_file_tuples = [(f, False) for f in discovery["sim002_train"]]
    if include_sim004:
        train_file_tuples.extend([(f, False) for f in discovery["sim004_train"]])
    if include_sim005:
        train_file_tuples.extend([(f, True) for f in discovery["sim005_train"]])

    val_file_tuples = [(f, False) for f in discovery["sim002_val"]]
    test_file_tuples = [(f, False) for f in discovery["sim002_test"]]

    counts = count_dataset_trajectories(discovery)
    print("\nExact Trajectory Inventory:")
    print(f"  TRAIN trajectories: {counts['train_trajectories']:,}")
    print(f"  VAL trajectories:   {counts['val_trajectories']:,}")
    print(f"  TEST trajectories:  {counts['test_trajectories']:,}")

    # Helper to stream a file list into memmap
    def stream_split_to_memmap(file_tuples, n_trajectories, split_name):
        print(f"\nBuilding {split_name} cache ({n_trajectories:,} trajectories)...")
        total_rows = n_trajectories * TRAJECTORY_LENGTH
        states_path = cache_dir / f"{split_name}_states.dat"
        inputs_path = cache_dir / f"{split_name}_inputs.dat"

        # Temporary files for atomic creation
        tmp_states_path = cache_dir / f"{split_name}_states.dat.tmp"
        tmp_inputs_path = cache_dir / f"{split_name}_inputs.dat.tmp"

        x_mmap = np.memmap(tmp_states_path, dtype=np.float32, mode="w+", shape=(total_rows, 10))
        u_mmap = np.memmap(tmp_inputs_path, dtype=np.float32, mode="w+", shape=(total_rows, 2))

        curr_traj = 0
        t_split = time.time()
        for idx, (fpath, is_s5) in enumerate(file_tuples):
            x_batch, u_batch, n_batch = parse_single_pkl_file(fpath, is_sim005=is_s5)
            if n_batch == 0:
                continue

            r_start = curr_traj * TRAJECTORY_LENGTH
            r_end = (curr_traj + n_batch) * TRAJECTORY_LENGTH

            x_mmap[r_start:r_end] = x_batch.reshape(-1, 10)
            u_mmap[r_start:r_end] = u_batch.reshape(-1, 2)
            curr_traj += n_batch

            if (idx + 1) % 500 == 0 or (idx + 1) == len(file_tuples):
                pct = (idx + 1) / len(file_tuples) * 100
                print(
                    f"  {split_name}: {idx + 1}/{len(file_tuples)} files ({pct:.1f}%) "
                    f"| {curr_traj:,}/{n_trajectories:,} trajectories [{time.time() - t_split:.1f}s]"
                )

        if curr_traj != n_trajectories:
            raise RuntimeError(
                f"Trajectory count mismatch during {split_name} cache build: expected {n_trajectories}, wrote {curr_traj}"
            )

        x_mmap.flush()
        u_mmap.flush()
        del x_mmap
        del u_mmap

        # Atomic rename
        if states_path.exists():
            os.remove(states_path)
        if inputs_path.exists():
            os.remove(inputs_path)
        os.replace(tmp_states_path, states_path)
        os.replace(tmp_inputs_path, inputs_path)
        print(f"  -> Saved {states_path.name} ({os.path.getsize(states_path) / 1e6:.1f} MB)")
        print(f"  -> Saved {inputs_path.name} ({os.path.getsize(inputs_path) / 1e6:.1f} MB)")

    # Build all 3 splits
    stream_split_to_memmap(train_file_tuples, counts["train_trajectories"], "TRAIN")
    stream_split_to_memmap(val_file_tuples, counts["val_trajectories"], "VAL")
    stream_split_to_memmap(test_file_tuples, counts["test_trajectories"], "TEST")

    # Generate manifest
    manifest = {
        "CACHE_SCHEMA_VERSION": CACHE_SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_dir": str(dataset_dir),
        "trajectory_length": TRAJECTORY_LENGTH,
        "state_order": STATE_ORDER,
        "input_order": INPUT_ORDER,
        "c1_index": 9,
        "file_counts": {
            "sim002_train": len(discovery["sim002_train"]),
            "sim002_val": len(discovery["sim002_val"]),
            "sim002_test": len(discovery["sim002_test"]),
            "sim004_train": len(discovery["sim004_train"]),
            "sim005_train": len(discovery["sim005_train"]),
            "total_files": discovery["total_files"],
        },
        "trajectory_counts": counts,
        "shapes": {
            "TRAIN_states": [counts["train_trajectories"] * TRAJECTORY_LENGTH, 10],
            "TRAIN_inputs": [counts["train_trajectories"] * TRAJECTORY_LENGTH, 2],
            "VAL_states": [counts["val_trajectories"] * TRAJECTORY_LENGTH, 10],
            "VAL_inputs": [counts["val_trajectories"] * TRAJECTORY_LENGTH, 2],
            "TEST_states": [counts["test_trajectories"] * TRAJECTORY_LENGTH, 10],
            "TEST_inputs": [counts["test_trajectories"] * TRAJECTORY_LENGTH, 2],
        },
    }

    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_manifest, manifest_path)
    print(f"\nManifest successfully written to {manifest_path}")

    return manifest


def validate_cache(cache_dir: Path, expected_dataset_dir: Optional[Path] = None) -> Tuple[bool, Optional[Dict]]:
    """
    Validate that existing cache files match schema and are structurally sound.
    """
    cache_dir = Path(cache_dir).resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return False, None

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if manifest.get("CACHE_SCHEMA_VERSION") != CACHE_SCHEMA_VERSION:
            print(f"Cache schema version mismatch: {manifest.get('CACHE_SCHEMA_VERSION')} != {CACHE_SCHEMA_VERSION}")
            return False, manifest

        shapes = manifest.get("shapes", {})
        for name, expected_shape in shapes.items():
            dat_path = cache_dir / f"{name}.dat"
            if not dat_path.exists():
                print(f"Missing cache data file: {dat_path}")
                return False, manifest

            expected_bytes = expected_shape[0] * expected_shape[1] * 4  # float32 = 4 bytes
            actual_bytes = os.path.getsize(dat_path)
            if actual_bytes != expected_bytes:
                print(f"Size mismatch for {dat_path.name}: expected {expected_bytes}, got {actual_bytes}")
                return False, manifest

        return True, manifest
    except Exception as e:
        print(f"Cache validation error: {e}")
        return False, None


def load_cached_split(
    cache_dir: Path, split_name: str, mode: str = "r"
) -> Tuple[np.memmap, np.memmap, Dict]:
    """
    Load memory-mapped arrays for a given split (TRAIN, VAL, TEST).
    """
    cache_dir = Path(cache_dir).resolve()
    valid, manifest = validate_cache(cache_dir)
    if not valid:
        raise RuntimeError(f"Cache in {cache_dir} is invalid or incomplete. Run build_memmap_cache first.")

    s_shape = tuple(manifest["shapes"][f"{split_name}_states"])
    i_shape = tuple(manifest["shapes"][f"{split_name}_inputs"])

    s_path = cache_dir / f"{split_name}_states.dat"
    i_path = cache_dir / f"{split_name}_inputs.dat"

    x_mmap = np.memmap(s_path, dtype=np.float32, mode=mode, shape=s_shape)
    u_mmap = np.memmap(i_path, dtype=np.float32, mode=mode, shape=i_shape)

    return x_mmap, u_mmap, manifest


def main():
    parser = argparse.ArgumentParser(description="Dataset Ingestion & Memory-Mapped Cache Pipeline")
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR), help="Path to OHSUSimPopulation")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR), help="Path to cache directory")
    parser.add_argument("--rebuild", action="store_true", help="Force cache rebuild")
    parser.add_argument("--verify-only", action="store_true", help="Verify discovery and cache without modifying")
    args = parser.parse_args()

    ds_dir = Path(args.dataset_dir)
    c_dir = Path(args.cache_dir)

    if args.verify_only:
        print("Discovering dataset...")
        discovery = discover_population_dataset(ds_dir)
        print(f"Discovered {discovery['total_files']} files across SIM002/004/005.")
        valid, manifest = validate_cache(c_dir)
        print(f"Cache validation status in {c_dir}: {'VALID' if valid else 'INVALID / NOT FOUND'}")
        if valid:
            print("Manifest trajectory counts:", manifest.get("trajectory_counts"))
    else:
        build_memmap_cache(ds_dir, c_dir, force_rebuild=args.rebuild)


if __name__ == "__main__":
    main()
