# Population-Level Neural State-Space Model Training

This repository reproduces the population-level training pipeline from:
**A Physiologically-Constrained Neural Network Digital Twin Framework for Replicating Glucose Dynamics in Type 1 Diabetes** (Roquemen-Echeverri et al., arXiv:2508.05705).

## What This Reproduces
- **End-to-End Population NN Training**: Reads the generated mechanistic `DP_train` and trains the `CGMOHSUSimStateSpaceModel_V2` using exact paper hyper-parameters, state alignments, robust scaling rules, non-linear physical constraints (Eq. 5-9), and 5-hour trajectory optimization.
- **Evaluation**: Performs paired Two One-Sided Tests (TOST) against the simulated ground truth on the held-out `DP_test`. 

## What This Does NOT Reproduce
- **Data Generation**: The underlying mechanistic `.mat` files are immutable upstream artifacts.
- **Bayesian Architecture Search**: We assume the final published network topology from the authors.
- **Formal Conformal Verification (Algorithm 2)**: The MILP/Gurobi monotonic verification logic is not implemented in this repository.

## Known Upstream Limitations
- **Day-ID Leakage**: The fixed upstream generator assigned dataset splits by evaluating the identity of the entire sorted 7-day tuple, rather than individual daily scenarios. Consequently, individual meal `day_ids` may leak across the Train/Val/Test boundaries. The downstream training scripts inherit this physical artifact directly to maintain experimental integrity.

## Scientific Configuration
The production configuration is strictly defined based on the paper or official artifacts:
- **PAPER-EXPLICIT**: `SEQ_LEN = 61`, `TRAIN_OVERLAP = 0.75`, `BATCH_SIZE = 128`, `LR = 1e-3`, `ALPHA = 0.7`, `BETA = 0.08`, `LR_DECAY_PER_EPOCH = np.exp(-0.1)`
- **RECONSTRUCTION / OFFICIAL ARTIFACTS**: 
  - `MAX_EPOCHS = 15`
  - `PATIENCE = None`
  - `VAL_OVERLAP = 0.0`
  - `TEST_OVERLAP = 0.0`
  - `CHECKPOINT_POLICY = "final_epoch"`

## Pre-Generated Datasets
The population dataset is already generated, fixed, and must not be regenerated/resplit for this reproduction. `merge_parts.py` is included for provenance/reconstruction only.

## MOHESR HPC Usage
The repository is configured for the MOHESR HPC gpu5 partition. The environment expects:
- Python 3.11.16
- PyTorch 1.13.0+cu117
- CUDA runtime 11.7
- h5py 3.16.0
- numpy 1.23.5
- scikit-learn 1.2.2
- scipy 1.15.3
- joblib 1.6.0
- The `T1DSim_AI` package (pinned submodule, editable install)

### Paths & Artifacts
- **Dataset**: `/tmp/cugp012/population_development_dataset_merged.mat`
- **Cache**: `/tmp/cugp012/cache`
- **Reference Models**: `models/PopulationModel/` (Preserved legacy reference results)
- **Production Outputs**: `models/PopulationModel_v2/` (Outputs of the new final HPC reproduction)

### 1. Prepare Cache
To safely build the memmap cache, validate the dataset SHA-256, and dump a verified `cache_manifest.json` before training:
```bash
sbatch hpc_prepare_cache.sh
```

### 2. Benchmark Run
To accurately measure the real forward/backward timing per batch without updating the optimizer:
```bash
sbatch hpc_benchmark.sh
```

### 3. Fresh Production Run
To start the full 15-epoch population training from scratch:
```bash
sbatch hpc_train.sh
```

### 4. Resumed Production Run
To safely resume training from the exact state saved at the last completed epoch boundary:
```bash
sbatch --export=ALL,RESUME_CHECKPOINT=/nfs/slurm/cugp012/training_population/models/PopulationModel_v2/training_state_latest.pt hpc_train.sh
```

### Slurm Logs
Logs are saved as `cache_%j.out`, `cache_%j.err`, `bench_%j.out`, `bench_%j.err`, `train_%j.out`, `train_%j.err` in the repository root.
