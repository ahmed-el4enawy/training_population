#!/bin/bash
#SBATCH --job-name=t1d_train
#SBATCH --partition=gpu5
#SBATCH --gres=gpu:a100_1g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --chdir=/nfs/slurm/cugp012/training_population

echo "Hostname: $(hostname)"
echo "Date: $(date)"
nvidia-smi

/nfs/slurm/cugp012/envs/t1d/bin/python -c "import sys; print('Python version:', sys.version)"
/nfs/slurm/cugp012/envs/t1d/bin/python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA availability:', torch.cuda.is_available()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Resume check if variable is set in the environment
if [ ! -z "$RESUME_CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
fi

/nfs/slurm/cugp012/envs/t1d/bin/python -u train_population_model.py
