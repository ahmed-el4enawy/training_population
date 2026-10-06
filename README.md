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

## Unresolved Scientific Choices
The paper omits certain explicit population-level constants. To enforce scientific rigor, you **must explicitly select** these values inside `train_population_model.py` and `evaluate_population_model.py` before running:
1. `MAX_EPOCHS`: The population training epoch limit (the paper's `150` applies to individual-level models, and official code checkpoints suggest `15`).
2. `PATIENCE`: Optional early stopping.
3. `CHECKPOINT_POLICY`: Whether to use the `final_epoch` or `best_validation` weights.
4. `VAL_OVERLAP`: Stride for validation sequences (BayesOpt states 0 overlap, final model is ambiguous).
5. `TEST_OVERLAP`: Stride for the final test set evaluations.

## Usage

### 1. Clone and Initialize
```bash
git clone https://github.com/ahmed-el4enawy/training_population.git
cd training_population
git submodule update --init
```

### 2. Environment Setup
```bash
pip install -r requirements.txt
# If T1DSim_AI is not in your python path, you can install it in editable mode:
pip install -e ./T1DSim_AI
```

### 3. Pipeline
1. **Merge Data**: (Produces `population_development_dataset_merged.mat` with explicit splits)
   ```bash
   python merge_parts.py
   ```
2. **Train Population Model**: 
   ```bash
   python train_population_model.py
   ```
3. **Evaluate**: 
   ```bash
   python evaluate_population_model.py
   ```

## MOHESR HPC Usage

The repository is configured for the MOHESR HPC gpu5 partition.

### Paths
- **Dataset**: `/tmp/cugp012/population_development_dataset_merged.mat`
- **Cache**: `/tmp/cugp012/cache`
- **Outputs**: `/nfs/slurm/cugp012/training_population/models/PopulationModel_v2/`

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
