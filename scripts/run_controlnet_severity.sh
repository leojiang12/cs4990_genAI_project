#!/bin/bash
#SBATCH --job-name=controln
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

echo "===== JOB START $(date) ====="

# 1) Activate conda (preserves SLURM's CUDA_VISIBLE_DEVICES)
source /data03/home/leojiang/miniconda3/etc/profile.d/conda.sh
conda activate xbd

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
python - <<<'import torch; print(\"torch.cuda.device_count()=\", torch.cuda.device_count())'

# 2) Prepare TensorBoard logdir (only rank0 will write to it)
TB_DIR="${SLURM_SUBMIT_DIR}/tb_logs"
mkdir -p "$TB_DIR"

# 3) Launch distributed training
torchrun --nproc_per_node=2 -m src.train_controlnet_severity_ddp \
    --labels_dir       data/train/labels \
    --images_dir       data/train/images \
    --batch_size       2 \
    --lr               1e-4 \
    --epochs           10 \
    --ckpt_dir         checkpoints \
    --tensorboard_dir  "$TB_DIR"

echo "===== JOB END $(date) ====="
