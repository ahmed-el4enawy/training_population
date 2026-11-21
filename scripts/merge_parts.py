"""
merge_parts.py

Merges T1DSim_population_part1..4.mat (written by generate_population_dataset.m)
into ONE file, population_development_dataset_merged.mat, in exactly the layout
train_population_model.py reads:

    dataset_states  (N, T, 10)   state order [Q1,Q2,S1,S2,I,X1,X2,X3,C2,C1]
    dataset_inputs  (N, T, 2)    [u_I (U/hr), u_carbs (g)]
    dataset_glucose (N, T)       CGM in mg/dL
    dataset_split_id(N,)         0=Train, 1=Val, 2=Test
    dataset_day_ids (N, 7)       Daily scenario IDs
    dataset_is_rare (N,)         Boolean flag for rare traces
    dataset_gid     (N,)         Reconstructed group ID

This is a STRUCTURAL MERGER ONLY.
No GROUP=5 padding, no rare dropping, no arbitrary re-splitting.
"""
import argparse
import os
import numpy as np
import h5py

# MATLAB channel index (0-based) for each entry of [Q1,Q2,S1,S2,I,X1,X2,X3,C2,C1]
# MATLAB order: [S1,S2,I,X1,X2,X3,Q1,Q2,C1,C2] -> idx 0..9
STATE_PERM = [6, 7, 0, 1, 2, 3, 4, 5, 9, 8]
CHUNK = 500  # traces per read/write block

SPLIT_MAP = {"D_train": 0, "D_val": 1, "D_test": 2}

def find_root(f):
    """Return the h5 group prefix that contains D_train/D_val/D_test."""
    if "dataset" in f and "D_train" in f["dataset"]:
        return "dataset/"
    if "D_train" in f:
        return ""
    raise RuntimeError("Could not find D_train/D_val/D_test in file")

def plan_split(f, root, split):
    """Extract information for the split without dropping or padding traces."""
    g = f[f"{root}{split}"]
    n = g["x"].shape[0]  # on-disk (n,10,T)
    if n == 0:
        return np.zeros(0, dtype=np.int64), None, None, None
    
    # In MATLAB v7.3, HDF5 arrays are reversed (e.g. 7xN becomes Nx7)
    day_ids = np.asarray(g["day_ids"][()]).reshape(n, -1)  # (n,7)
    is_rare = np.asarray(g["is_rare"][()]).reshape(-1).astype(bool)
    
    # Reconstruct gid based on identical 7-day sorted tuples
    keys = np.sort(day_ids, axis=1)
    _, gid = np.unique(keys, axis=0, return_inverse=True)
    gid = np.asarray(gid).reshape(-1).astype(np.int32)
    
    idx = np.arange(n, dtype=np.int64)
    # stable sort by group id, non-rare before rare, so traces are contiguous
    order = np.lexsort((is_rare[idx], gid[idx]))
    idx = idx[order]
    
    return idx, day_ids[order], is_rare[order], gid[order]

def read_rows(ds, rows):
    uniq = np.unique(rows)
    block = ds[uniq]
    return block[np.searchsorted(uniq, rows)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--prefix", default="T1DSim_population_part")
    ap.add_argument("--out", default="population_development_dataset_merged.mat")
    args = ap.parse_args()

    part_files = [os.path.join(args.dir, f"{args.prefix}{i}.mat") for i in range(1, args.parts + 1)]
    for p in part_files:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    out_path = os.path.join(args.dir, args.out)

    # ---- Pass 1: plan order + total count ----
    plans = []
    total, T, n_days = 0, None, 7
    
    for p in part_files:
        with h5py.File(p, "r") as f:
            root = find_root(f)
            for split in ["D_train", "D_val", "D_test"]:
                idx, d_ids, i_rare, g_ids = plan_split(f, root, split)
                if T is None and len(idx) > 0:
                    T = f[f"{root}{split}"]["x"].shape[2]
                plans.append((p, root, split, idx, d_ids, i_rare, g_ids))
                total += len(idx)
                print(f"{os.path.basename(p)} {split}: {len(idx)} traces")
    print(f"Total traces: {total} | T = {T}")

    # ---- Create output (MATLAB v7.3 layout) ----
    with h5py.File(out_path, "w", userblock_size=512) as out:
        d_states = out.create_dataset("dataset_states", (10, T, total), dtype="float32", chunks=(10, T, 8))
        d_inputs = out.create_dataset("dataset_inputs", (2, T, total), dtype="float32", chunks=(2, T, 8))
        d_gluc = out.create_dataset("dataset_glucose", (T, total), dtype="float32", chunks=(T, 8))
        
        d_split = out.create_dataset("dataset_split_id", (total,), dtype="int32")
        d_dayids = out.create_dataset("dataset_day_ids", (n_days, total), dtype="int64")
        d_rare = out.create_dataset("dataset_is_rare", (total,), dtype="bool")
        d_gid = out.create_dataset("dataset_gid", (total,), dtype="int32")
        
        for d in (d_states, d_inputs, d_gluc):
            d.attrs.create("MATLAB_class", np.bytes_("single"))

        # ---- Pass 2: stream, reorder, write ----
        pos = 0
        train_days, val_days, test_days = set(), set(), set()
        
        for p, root, split, idx, d_ids, i_rare, g_ids in plans:
            if len(idx) == 0:
                continue
            split_id = SPLIT_MAP[split]
            
            # Record day_ids for leakage diagnostic
            flat_days = set(d_ids.flatten())
            if split_id == 0: train_days.update(flat_days)
            elif split_id == 1: val_days.update(flat_days)
            elif split_id == 2: test_days.update(flat_days)
            
            with h5py.File(p, "r") as f:
                g = f[f"{root}{split}"]
                for s in range(0, len(idx), CHUNK):
                    rows = idx[s:s + CHUNK]
                    c = len(rows)
                    x = read_rows(g["x"], rows)
                    u = read_rows(g["u"], rows)
                    cg = read_rows(g["cgm"], rows)
                    x = x[:, STATE_PERM, :]
                    
                    d_states[:, :, pos:pos + c] = np.transpose(x, (1, 2, 0)).astype(np.float32)
                    d_inputs[:, :, pos:pos + c] = np.transpose(u, (1, 2, 0)).astype(np.float32)
                    d_gluc[:, pos:pos + c] = np.transpose(cg, (1, 0)).astype(np.float32)
                    
                    d_split[pos:pos + c] = split_id
                    d_dayids[:, pos:pos + c] = np.transpose(d_ids[s:s + c], (1, 0))
                    d_rare[pos:pos + c] = i_rare[s:s + c]
                    d_gid[pos:pos + c] = g_ids[s:s + c]
                    
                    pos += c
            print(f"  wrote {pos}/{total}")
            
    print("\n--- Leakage Diagnostic ---")
    train_val_leak = train_days.intersection(val_days)
    train_test_leak = train_days.intersection(test_days)
    val_test_leak = val_days.intersection(test_days)
    
    print(f"Train ∩ Val leaked day_ids:  {len(train_val_leak)}")
    print(f"Train ∩ Test leaked day_ids: {len(train_test_leak)}")
    print(f"Val ∩ Test leaked day_ids:   {len(val_test_leak)}")
    
    if len(train_val_leak) > 0 or len(train_test_leak) > 0 or len(val_test_leak) > 0:
        print("WARNING: UPSTREAM DATASET LIMITATION DETECTED. Individual day_ids leaked across splits!")

    # ---- Add MATLAB v7.3 header ----
    hdr = b"MATLAB 7.3 MAT-file, Platform: PCWIN64, Created by merge_parts.py"
    hdr = hdr.ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02" + b"IM"
    with open(out_path, "r+b") as fh:
        fh.write(hdr.ljust(512, b"\x00"))

    print(f"\nDone: {out_path}")

if __name__ == "__main__":
    main()
