# Population-Level Neural State-Space Model Training

This repository contains the training and evaluation pipeline for the population-level digital twin model $T1DSim_{NN}^P$ (`CGMOHSUSimStateSpaceModel_V2` / `PackedPopulationModel`), reproducing the methodology from:
**"A Physiologically-Constrained Neural Network Digital Twin Framework for Replicating Glucose Dynamics in Type 1 Diabetes"** (Roquemen-Echeverri et al., arXiv:2508.05705).

---

## Architecture & Scientific Design

- **State Space (10 states)**:
  `[Q1, Q2, S1, S2, I, X1, X2, X3, C2, C1]`
  - $Q_1$: Continuous Glucose Monitoring / blood glucose in mg/dL (`output_cgm`).
  - $Q_2$: Interstitial non-accessible glucose compartment (`state_Q2`).
  - $S_1, S_2$: Subcutaneous insulin absorption chain (`state_S1, state_S2`).
  - $I$: Plasma insulin concentration (`state_I`).
  - $X_1, X_2, X_3$: Insulin action on distribution, disposal, and endogenous production (`state_X1, state_X2, state_X3`).
  - $C_2$: Carbohydrate absorption rate from gut into systemic circulation (`state_Ug`).
  - $C_1$: Second gut absorption compartment (`state_Ug2 = 0.0` per authors' official dataset schema; masked in consistency loss).
- **Inputs (2 channels)**:
  - $u_0$: Total insulin delivery rate in $\text{U/h}$ (`input_insulin`).
  - $u_1$: Prandial carbohydrate intake in $\text{g}$ (`input_meal_carbs`).
- **Mathematical Simulation**:
  - Forward Euler numerical integration with step $\Delta t = 5.0\text{ min}$.
- **Physiological Loss Function (Eq. 5–9)**:
  - Asymmetric penalty: $2\times$ multiplier for under-predicting hypoglycemia ($\le 70\text{ mg/dL}$), $6\times$ multiplier for over-predicting hyperglycemia ($\ge 180\text{ mg/dL}$).
  - State consistency penalty: $\alpha = 0.7$, state weights $w = [5/24, 1/6, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/24]$ with $C_1$ masked out.
  - Soft barrier constraint: $\beta = 0.08$ penalizing states below empirical physical floor.
- **Hyperparameters (Paper Explicit)**:
  - $\text{SEQ\_LEN} = 61$ (5-hour rollout windows).
  - Training window candidate overlap = $75\%$ ($\text{hop} = 15$ steps).
  - Batch size = $128$.
  - Initial learning rate = $10^{-3}$, decayed per epoch by $e^{-0.1} \approx 0.9048$.
  - Training epochs = $15$.
  - Training sequence pool cap = $1,000,000$ unique sequences (matching the paper's reported training scale).

---

## Dataset Ingestion & Caching Pipeline (`dataset_pipeline.py`)

The pipeline ingests the complete 5,010 `.pkl` reproduced population dataset from `CodeForGeneratingDataset/GeneratedDataset/OHSUSimPopulation/`:

1. **Split Structure**:
   - **SIM002 Baseline ($\text{delay} = 0\text{ min}$)**:
     - `TRAIN`: 18 subjects $\times$ 15 files = 270 files (2,970 trajectories).
     - `VALIDATION`: 5 subjects $\times$ 15 files = 75 files (825 trajectories).
     - `TEST`: 5 subjects $\times$ 15 files = 75 files (825 trajectories).
   - **SIM004 Prandial Delay ($5\text{ to }45\text{ min}$)**:
     - 9 delay conditions $\times$ 420 files = 3,780 files (41,580 trajectories) $\rightarrow$ Assigned to `TRAIN`.
   - **SIM005 Correction Boluses ($\text{CF} \in \{28, 35, 42\}$)**:
     - 810 files filtered to retain trajectories with $\text{correction\_bolus} > 0$ per author's methodology $\rightarrow$ Assigned to `TRAIN`.
2. **Subject-Level Isolation**:
   - Held-out test subjects (`042006, 042007, 042011, 042012, 042013`) and validation subjects are strictly isolated.
3. **Memory-Mapped Binary Caching**:
   - Pre-allocates and streams data into fast, zero-RAM-bloat binary files: `TRAIN_states.dat`, `TRAIN_inputs.dat`, `VAL_states.dat`, `VAL_inputs.dat`, `TEST_states.dat`, `TEST_inputs.dat`.
   - Generates a verified `manifest.json` with schema versioning and shape validation.

---

## Local Setup & Execution

### 1. Environment Setup
```bash
# Recommended Python version: 3.11
python -m venv .venv
.\.venv\Scripts\activate

# Install required dependencies
pip install torch numpy scipy scikit-learn pandas h5py
```

### 2. Run Test Suite
```bash
python -m unittest discover tests
```
The test suite covers:
- Dataset discovery, schema verification, column mapping, and subject-level split isolation.
- Window framing (5 hours, 75% overlap, hop=15) and trajectory-boundary safety.
- Memory-mapped cache creation, validation, and invalidation.
- Model checkpoint saving, loading, and resumption.
- Packed population model numerical and gradient equivalence.
- Clinical metrics (TIR, TAR, TBR, LBGI, HBGI, MG) and TOST statistical equivalence.

### 3. Build / Verify Dataset Cache
```bash
# Verify dataset discovery without writing:
python dataset_pipeline.py --verify-only

# Build full memory-mapped cache:
python dataset_pipeline.py
```

### 4. Evaluate Held-Out SIM002 TEST Cohort
```bash
python evaluate_population_model.py --checkpoint models/PopulationModel/population_model_05022024_epoch_15.pt
```

### 5. Training Execution
```bash
# Local test or benchmark run:
python train_population_model.py --benchmark-only

# Resuming training from an epoch boundary:
python train_population_model.py --resume models/PopulationModel_v2/training_state_latest.pt
```

---

## HPC Slurm Execution

Configured for GPU partitions (`gpu5` on MOHESR Slurm cluster):

```bash
# 1. Build and validate cache
sbatch hpc_prepare_cache.sh

# 2. Run benchmark to measure throughput per batch
sbatch hpc_benchmark.sh

# 3. Launch 15-epoch population training
sbatch hpc_train.sh

# 4. Resume interrupted training
sbatch --export=ALL,RESUME_CHECKPOINT=/nfs/slurm/cugp012/training_population/models/PopulationModel_v2/training_state_latest.pt hpc_train.sh
```
