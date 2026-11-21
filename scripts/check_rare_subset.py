import argparse
import sys
import numpy as np
import h5py
from train_population_model import get_splits

def print_split_stats(name, selected_idx, orig_idx, is_rare_full):
    selected = len(selected_idx)
    orig = len(orig_idx)
    
    if selected == 0 or orig == 0:
        print(f"{name}\n  [Empty Split]\n")
        return 0, 0, 0, 0
        
    selected_rare = int(np.sum(is_rare_full[selected_idx]))
    orig_rare = int(np.sum(is_rare_full[orig_idx]))
    
    selected_rare_pct = (selected_rare / selected * 100) if selected > 0 else 0
    orig_rare_pct = (orig_rare / orig * 100) if orig > 0 else 0
    retained_pct = (selected_rare / orig_rare * 100) if orig_rare > 0 else 0
    
    print(f"{name}")
    print(f"  selected traces          {selected}")
    print(f"  selected non-rare        {selected - selected_rare}")
    print(f"  selected rare            {selected_rare}")
    print(f"  selected rare percent    {selected_rare_pct:.2f}%")
    print(f"  original traces          {orig}")
    print(f"  original rare            {orig_rare}")
    print(f"  original rare percent    {orig_rare_pct:.2f}%")
    print(f"  rare retained            {retained_pct:.2f}%\n")
    
    return selected, selected_rare, orig, orig_rare

def main():
    parser = argparse.ArgumentParser(description="Verify rare-event retention in paper-scale subset")
    parser.add_argument("--dataset", type=str, default="/tmp/cugp012/population_development_dataset_merged.mat",
                        help="Path to the merged dataset MAT file")
    args = parser.parse_args()
    
    dataset_path = args.dataset
    
    try:
        with h5py.File(dataset_path, "r") as f:
            if "dataset_is_rare" not in f:
                raise ValueError("dataset_is_rare array is missing from the dataset.")
            if "dataset_split_id" not in f:
                raise ValueError("dataset_split_id array is missing from the dataset.")
            if "dataset_day_ids" not in f:
                raise ValueError("dataset_day_ids array is missing from the dataset.")
                
            is_rare = np.asarray(f["dataset_is_rare"][()]).reshape(-1)
            split_id = np.asarray(f["dataset_split_id"][()]).reshape(-1)
            day_ids = np.asarray(f["dataset_day_ids"][()])
            if day_ids.shape[0] == 7:
                day_ids = day_ids.T
                
            n_traces = len(split_id)
            if len(is_rare) != n_traces:
                raise ValueError(f"Shape inconsistency: split_id has {n_traces} elements but is_rare has {len(is_rare)}.")
            if day_ids.shape[0] != n_traces:
                raise ValueError(f"Shape inconsistency: split_id has {n_traces} elements but day_ids has {day_ids.shape[0]}.")
                
            # Treat is_rare as boolean
            is_rare = is_rare > 0
            
    except Exception as e:
        print(f"Failed to load and validate dataset: {e}")
        sys.exit(1)
        
    train_all = np.where(split_id == 0)[0]
    val_all = np.where(split_id == 1)[0]
    test_all = np.where(split_id == 2)[0]
    
    train_idx, val_idx, test_idx = get_splits(dataset_path)
    
    print("=" * 50)
    print("RARE-EVENT PREVALENCE VERIFICATION")
    print("=" * 50 + "\n")
    
    sel_t, sel_r_t, orig_t, orig_r_t = print_split_stats("TRAIN", train_idx, train_all, is_rare)
    sel_v, sel_r_v, orig_v, orig_r_v = print_split_stats("VAL", val_idx, val_all, is_rare)
    sel_te, sel_r_te, orig_te, orig_r_te = print_split_stats("TEST", test_idx, test_all, is_rare)
    
    total_sel = sel_t + sel_v + sel_te
    total_sel_rare = sel_r_t + sel_r_v + sel_r_te
    total_orig = orig_t + orig_v + orig_te
    total_orig_rare = orig_r_t + orig_r_v + orig_r_te
    
    total_sel_pct = (total_sel_rare / total_sel * 100) if total_sel > 0 else 0
    total_orig_pct = (total_orig_rare / total_orig * 100) if total_orig > 0 else 0
    total_retained_pct = (total_sel_rare / total_orig_rare * 100) if total_orig_rare > 0 else 0
    
    print("TOTAL")
    print(f"  selected traces          {total_sel}")
    print(f"  selected non-rare        {total_sel - total_sel_rare}")
    print(f"  selected rare            {total_sel_rare}")
    print(f"  selected rare percent    {total_sel_pct:.2f}%")
    print(f"  original traces          {total_orig}")
    print(f"  original rare            {total_orig_rare}")
    print(f"  original rare percent    {total_orig_pct:.2f}%")
    print(f"  rare retained            {total_retained_pct:.2f}%\n")
    
    diff_pct = total_sel_pct - total_orig_pct
    
    print("INTERPRETATION:")
    if abs(diff_pct) < 1.0:
        print(f"Rare-event prevalence preserved closely (Difference: {diff_pct:+.2f} percentage points)")
    else:
        print(f"Rare-event prevalence changed materially in the paper-scale subset (Difference: {diff_pct:+.2f} percentage points)")

if __name__ == "__main__":
    main()
