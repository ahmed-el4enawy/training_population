#!/bin/bash
#SBATCH --job-name=t1d_cache
#SBATCH --partition=gpu5
#SBATCH --gres=gpu:a100_1g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=cache_%j.out
#SBATCH --error=cache_%j.err
#SBATCH --chdir=/nfs/slurm/cugp012/training_population

set -euo pipefail

echo "Hostname: $(hostname)"
echo "Date: $(date)"

export PREPARE_CACHE_ONLY="1"

/nfs/slurm/cugp012/envs/t1d/bin/python -u train_population_model.py
