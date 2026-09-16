"""
merge_to_single_mat.py

Merges the 4 population_development_dataset_partN.mat files into ONE .mat
file (not HDF5-with-a-different-extension - an actual .mat, just written in
MATLAB's v7.3 format so it can hold >2GB variables, which plain v7/v5 .mat
cannot).

Requires:
    pip install scipy h5py hdf5storage numpy

RUN IN THIS ORDER
------------------
1) INSPECT_ONLY = True  -> prints part 1's structure. Confirm/adjust before
   merging 7GB of data.
2) INSPECT_ONLY = False -> merges all 4 parts in memory and writes one .mat.

MEMORY NOTE: this version loads all 4 parts fully into RAM to concatenate
(needs roughly 2x the total dataset size free, so ~14GB+ for a 7GB dataset).
If that's not available on your machine, tell me and I'll give you a
streaming version instead (writes incrementally via h5py, like the earlier
merge_mat_parts.py script, and just relabels the output .h5 as .mat - v7.3
.mat files ARE HDF5 files under the hood, so this is a legitimate option
too).
"""

import os
import sys
import gc
import numpy as np
import scipy.io as sio

try:
    import h5py
except ImportError:
    h5py = None

try:
    import hdf5storage
except ImportError:
    hdf5storage = None

# ==============================================================================
# CONFIG - EDIT THESE
# ==============================================================================
# NOTE: Update INPUT_DIR below to wherever the 4 part files actually live on
# your D: drive if it's not this folder.
INPUT_DIR = r"C:\Users\pc\Downloads\population"
INPUT_FILES = [
    "population_development_dataset_part1.mat",
    "population_development_dataset_part2.mat",
    "population_development_dataset_part3.mat",
    "population_development_dataset_part4.mat",
]
OUTPUT_PATH = r"C:\Users\pc\Downloads\population\population_development_dataset_merged.mat"

INSPECT_ONLY = False  # <- merge will run immediately; set True first if you want to re-check structure

IGNORE_KEYS = {"__header__", "__version__", "__globals__"}


# ==============================================================================
# Generic loader: tries scipy (legacy MATLAB), falls back to h5py (v7.3/HDF5)
# ==============================================================================
def load_mat_flat(path):
    try:
        d = sio.loadmat(path, simplify_cells=True)
        return {k: v for k, v in d.items() if k not in IGNORE_KEYS}, "scipy"
    except NotImplementedError:
        if h5py is None:
            raise RuntimeError("This .mat is v7.3/HDF5 - run: pip install h5py")
        out = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                if k in IGNORE_KEYS:
                    continue
                arr = f[k][()]
                if isinstance(arr, np.ndarray) and arr.ndim > 1:
                    arr = arr.T  # v7.3 stores transposed vs. numpy row-major
                out[k] = arr
        return out, "h5py"


def inspect(path):
    print(f"\n=== Inspecting: {path} ===")
    d, backend = load_mat_flat(path)
    print(f"(read using: {backend})")
    print(f"Top-level keys ({len(d)}):")
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            print(f"  - {k!r:30s} shape={v.shape} dtype={v.dtype}")
        elif isinstance(v, dict):
            print(f"  - {k!r:30s} (nested dict, keys: {list(v.keys())[:10]})")
        elif isinstance(v, list):
            print(f"  - {k!r:30s} (list, len={len(v)}, elem type={type(v[0]) if v else None})")
        else:
            print(f"  - {k!r:30s} type={type(v)}")
    return d


# ==============================================================================
# MERGE (in-memory concatenation, then single v7.3 .mat write)
# ==============================================================================
def merge_and_save(input_paths, output_path):
    if hdf5storage is None:
        raise RuntimeError(
            "pip install hdf5storage  (needed to write >2GB .mat / v7.3 format)"
        )

    merged = {}
    for i, path in enumerate(input_paths):
        print(f"\n[{i+1}/{len(input_paths)}] Loading {path} ...")
        d, backend = load_mat_flat(path)
        print(f"  -> loaded via {backend}, {len(d)} variables")

        for key, arr in d.items():
            if not isinstance(arr, np.ndarray):
                print(f"  !! skipping non-array key '{key}' (type {type(arr)})")
                continue
            if key not in merged:
                merged[key] = [arr]
            else:
                merged[key].append(arr)

        del d
        gc.collect()

    print("\nConcatenating all parts per variable ...")
    for key in list(merged.keys()):
        merged[key] = np.concatenate(merged[key], axis=0)
        print(f"  '{key}': final shape {merged[key].shape}")

    # Make sure the destination folder exists before writing (D: drive folder
    # may not exist yet on a fresh machine).
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"\nWriting merged .mat (v7.3) to: {output_path}")
    hdf5storage.savemat(
        output_path,
        merged,
        format="7.3",
        matlab_compatible=True,
        oned_as="column",
    )
    print("Done.")


def main():
    input_paths = [os.path.join(INPUT_DIR, f) for f in INPUT_FILES]
    for p in input_paths:
        if not os.path.exists(p):
            print(f"ERROR: file not found: {p}")
            sys.exit(1)

    if INSPECT_ONLY:
        inspect(input_paths[0])
        print(
            "\n--- Inspection complete ---\n"
            "If these look like flat per-variable arrays (Q1, input_insulin, ...),\n"
            "set INSPECT_ONLY = False and re-run to merge.\n"
            "If it's a single key holding a struct array instead, tell me and\n"
            "I'll adjust merge_and_save() to handle that shape."
        )
        return

    merge_and_save(input_paths, OUTPUT_PATH)


if __name__ == "__main__":
    main()