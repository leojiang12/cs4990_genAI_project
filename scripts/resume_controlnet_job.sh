#!/bin/bash
#SBATCH --job-name=controln_resume
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/resume_train_%j.log
#SBATCH --error=logs/resume_train_%j.err

echo "===== JOB START $(date) ====="

# activate your conda env
source /data03/home/leojiang/miniconda3/etc/profile.d/conda.sh
conda activate xbd

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
python -c "import torch; print('GPUs available:', torch.cuda.device_count())"

# ensure log dirs exist
mkdir -p tb_logs

# call the resume wrapper
scripts/resume_controlnet_training.sh

echo "===== JOB END $(date) ====="
