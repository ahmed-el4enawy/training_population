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
  - `TRAIN_SEQUENCE_LIMIT = 1_000_000` (matches the paper's reported approximate training-sequence scale)
- **ENGINEERING ONLY (scientifically equivalent)**:
  - The 10 independent shallow subnetworks are packed into two masked dense operations per Euler step to reduce CUDA kernel-launch overhead. Inactive connections are permanently zero and receive zero gradients.
  - Production checkpoints are exported back to the official `CGMOHSUSimStateSpaceModel_V2` state-dict layout.
  - Final-training validation passes are disabled because Algorithm 1 defines optimization over `DP_train`; the paper's no-overlap validation pass is described for Bayesian architecture optimization. Held-out evaluation remains separate.

## Pre-Generated Datasets
The merged population dataset is already generated, fixed, and is **not regenerated or re-split** by training. The upstream generator produced a larger artifact than the paper-scale development set because it used `num_scenarios=46200` and then generated `variants_per_scenario=5` before rare-event supplementation. The paper reports **323,400 simulated days = 46,200 seven-day traces total**.

For training, the immutable merged artifact is therefore deterministically reduced **within the existing Train/Val/Test split assignments and by whole 7-day meal-scenario groups** to approximately the paper-scale 60/20/20 counts (27,720 / 9,240 / 9,240 traces). No data are regenerated and no trace crosses to another split.

The paper also states that the population model used **approximately one million 5-hour training sequences**. The training sampler deterministically caps its unique sequence pool at 1,000,000 while preserving the paper-explicit 5-hour length, 75% overlap candidate framing, batch size 128, and epoch reshuffling.

`merge_parts.py` remains provenance/reconstruction code only.

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
To measure a production-like forward/loss/backward/**Adam update** timing per batch:
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
