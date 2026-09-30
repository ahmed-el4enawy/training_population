"""
merge_parts.py

Merges T1DSim_population_part1..4.mat (written by generate_population_dataset.m)
into ONE file, population_development_dataset_merged.mat, in exactly the layout
train_population_model.py reads:

    dataset_states  (N, T, 10)   state order [Q1,Q2,S1,S2,I,X1,X2,X3,C2,C1]
    dataset_inputs  (N, T, 2)    [u_I (U/hr), u_carbs (g)]
    dataset_glucose (N, T)       CGM in mg/dL

No metrics are computed. Everything is streamed with h5py in small chunks
(RAM stays well under ~1 GB). Output is a MATLAB v7.3 (HDF5) file.

Key details handled for you:
  * The MATLAB generator stores states as [S1,S2,I,X1,X2,X3,Q1,Q2,C1,C2]; the
    trainer expects Q1 in channel 0 (it overwrites channel 0 with glucose), so
    states are REORDERED here. Without this, S1 would be destroyed.
  * The trainer groups every 5 consecutive rows as one meal pattern
    (5 initial-glucose variants). Traces are therefore sorted so that all
    traces sharing the same 7-day meal set are contiguous, and the total
    count of each group is a multiple of 5.
  * Rare-event traces (extra hypo/hyper re-simulations) break the "5 per
    group" rule. Default: DROP them (RARE_MODE="drop"). "keep" keeps them and
    pads each group to a multiple of 5 by repeating its last trace.

Usage:
    python merge_parts.py
    python merge_parts.py --dir C:\\Users\\pc\\Downloads\\population --rare keep
"""
import argparse
import os
import numpy as np
import h5py

# MATLAB channel index (0-based) for each entry of [Q1,Q2,S1,S2,I,X1,X2,X3,C2,C1]
# MATLAB order: [S1,S2,I,X1,X2,X3,Q1,Q2,C1,C2] -> idx 0..9
STATE_PERM = [6, 7, 0, 1, 2, 3, 4, 5, 9, 8]
GROUP = 5
CHUNK = 500  # traces per read/write block


def find_root(f):
    """Return the h5 group prefix that contains D_train/D_val/D_test."""
    if "dataset" in f and "D_train" in f["dataset"]:
        return "dataset/"
    if "D_train" in f:  # fallback save path in the generator
        return ""
    raise RuntimeError("Could not find D_train/D_val/D_test in file")


def plan_split(f, root, split, rare_mode):
    """Return an array of source indices (into that split) in the final order."""
    g = f[f"{root}{split}"]
    n = g["x"].shape[0]  # on-disk (n,10,T)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    day_ids = np.asarray(g["day_ids"][()]).reshape(n, -1)  # on-disk (n,7)
    is_rare = np.asarray(g["is_rare"][()]).reshape(-1).astype(bool)

    keys = np.sort(day_ids, axis=1)
    _, gid = np.unique(keys, axis=0, return_inverse=True)
    gid = np.asarray(gid).reshape(-1)

    idx = np.arange(n)
    if rare_mode == "drop":
        idx = idx[~is_rare]
    # stable sort by group id, non-rare before rare
    order = np.lexsort((is_rare[idx], gid[idx]))
    idx = idx[order]

    if rare_mode == "keep":
        out = []
        g_sorted = gid[idx]
        for grp in np.unique(g_sorted):
            members = idx[g_sorted == grp]
            pad = (-len(members)) % GROUP
            if pad:
                members = np.concatenate([members, np.repeat(members[-1], pad)])
            out.append(members)
        idx = np.concatenate(out) if out else idx
    else:
        # sanity: every group should have a multiple of 5 traces
        g_sorted = gid[idx]
        _, counts = np.unique(g_sorted, return_counts=True)
        bad = int(np.sum(counts % GROUP != 0))
        if bad:
            print(f"  WARNING: {bad} groups in {split} are not a multiple of {GROUP}")
    return idx


def read_rows(ds, rows):
    """Fancy-read arbitrary (possibly repeated / unsorted) rows along axis 0."""
    uniq = np.unique(rows)
    block = ds[uniq]
    return block[np.searchsorted(uniq, rows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--prefix", default="T1DSim_population_part")
    ap.add_argument("--out", default="population_development_dataset_merged.mat")
    ap.add_argument("--rare", choices=["drop", "keep"], default="drop")
    args = ap.parse_args()

    part_files = [os.path.join(args.dir, f"{args.prefix}{i}.mat") for i in range(1, args.parts + 1)]
    for p in part_files:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    out_path = os.path.join(args.dir, args.out)

    # ---- Pass 1: plan order + total count (metadata only, tiny reads) ----
    plans = []  # (part_file, root, split, src_index_array)
    total, T = 0, None
    for p in part_files:
        with h5py.File(p, "r") as f:
            root = find_root(f)
            for split in ("D_train", "D_val", "D_test"):
                idx = plan_split(f, root, split, args.rare)
                if T is None:
                    T = f[f"{root}{split}"]["x"].shape[2]
                plans.append((p, root, split, idx))
                total += len(idx)
                print(f"{os.path.basename(p)} {split}: {len(idx)} traces")
    print(f"Total traces: {total} | T = {T} | groups of {GROUP}: {total // GROUP}"
          f" | divisible by {GROUP}: {total % GROUP == 0}")

    # ---- Create output (MATLAB v7.3 layout: axes reversed on disk) ----
    with h5py.File(out_path, "w", userblock_size=512) as out:
        d_states = out.create_dataset("dataset_states", (10, T, total), dtype="float32", chunks=(10, T, 8))
        d_inputs = out.create_dataset("dataset_inputs", (2, T, total), dtype="float32", chunks=(2, T, 8))
        d_gluc = out.create_dataset("dataset_glucose", (T, total), dtype="float32", chunks=(T, 8))
        for d in (d_states, d_inputs, d_gluc):
            d.attrs.create("MATLAB_class", np.bytes_("single"))

        # ---- Pass 2: stream, reorder, write ----
        pos = 0
        for p, root, split, idx in plans:
            if len(idx) == 0:
                continue
            with h5py.File(p, "r") as f:
                g = f[f"{root}{split}"]
                for s in range(0, len(idx), CHUNK):
                    rows = idx[s:s + CHUNK]
                    c = len(rows)
                    x = read_rows(g["x"], rows)      # (c,10,T) MATLAB order
                    u = read_rows(g["u"], rows)      # (c,2,T)
                    cg = read_rows(g["cgm"], rows)   # (c,T)
                    x = x[:, STATE_PERM, :]          # -> [Q1,Q2,S1,S2,I,X1,X2,X3,C2,C1]
                    d_states[:, :, pos:pos + c] = np.transpose(x, (1, 2, 0)).astype(np.float32)
                    d_inputs[:, :, pos:pos + c] = np.transpose(u, (1, 2, 0)).astype(np.float32)
                    d_gluc[:, pos:pos + c] = np.transpose(cg, (1, 0)).astype(np.float32)
                    pos += c
            print(f"  wrote {pos}/{total}")

    # ---- Add MATLAB v7.3 header into the 512-byte userblock ----
    hdr = b"MATLAB 7.3 MAT-file, Platform: PCWIN64, Created by merge_parts.py"
    hdr = hdr.ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02" + b"IM"
    with open(out_path, "r+b") as fh:
        fh.write(hdr.ljust(512, b"\x00"))

    print(f"Done: {out_path}")
    print("State order: [Q1,Q2,S1,S2,I,X1,X2,X3,C2,C1] | inputs: [u_I(U/hr), u_carbs(g)]")


if __name__ == "__main__":
    main()
