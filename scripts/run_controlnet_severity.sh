#!/bin/bash
#SBATCH --job-name=controln
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH -t 3-00:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

echo "===== JOB START $(date) ====="

# 1) Purge & load the cluster’s CUDA modules:
module purge
module load cuda/11.7    # ↳ pick whatever your center uses
module load cudnn/8.2    # ↳ if required

# 2) Activate your conda env (preserves CUDA_VISIBLE_DEVICES):
source /data03/home/leojiang/miniconda3/etc/profile.d/conda.sh
conda activate xbd

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Using Python: $(which python)"
echo "Torch sees $(python - <<<'import torch;print(torch.cuda.device_count())') GPUs"

# 3) Prepare TensorBoard logs (only rank 0 will write)
TB_DIR="${SLURM_SUBMIT_DIR}/tb_logs"
mkdir -p "$TB_DIR"

# 4) Launch your DDP training directly under the conda env:
torchrun --nproc_per_node=2 -m src.train_controlnet_severity_ddp \
    --labels_dir       data/train/labels \
    --images_dir       data/train/images \
    --batch_size       2 \
    --lr               1e-4 \
    --epochs           10 \
    --ckpt_dir         checkpoints \
    --tensorboard_dir  "$TB_DIR"

echo "===== JOB END $(date) ====="
