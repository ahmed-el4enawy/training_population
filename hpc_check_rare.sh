#!/bin/bash
#SBATCH --job-name=check_rare
#SBATCH --partition=gpu5
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=check_rare_%j.out
#SBATCH --error=check_rare_%j.err
#SBATCH --chdir=/nfs/slurm/cugp012/training_population

set -euo pipefail

echo "Hostname: $(hostname)"
echo "Date: $(date)"

/nfs/slurm/cugp012/envs/t1d/bin/python -u check_rare_subset.py --dataset=/tmp/cugp012/population_development_dataset_merged.mat
