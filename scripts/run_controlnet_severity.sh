#!/bin/bash
#SBATCH --job-name=controln_sev
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 3-00:00:00

echo "========== JOB START $(date) =========="

# --- load conda and activate env ---
# adjust this path to your install
source /data03/home/leojiang/miniconda3/etc/profile.d/conda.sh
conda activate xbd

# --- start tensorboard in background ---
mkdir -p tb_logs
tensorboard --logdir tb_logs --bind_all --port=6006 &

# --- run distributed training ---
torchrun --nproc_per_node=2 -m src.train_controlnet_severity_ddp \
    --labels_dir data/train/labels \
    --images_dir data/train/images \
    --batch_size 2 \
    --lr 1e-4 \
    --epochs 10 \
    --ckpt_dir checkpoints \
    --log_dir tb_logs

echo "========== JOB  END  $(date) =========="
