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

# load your conda environment
source /data03/home/leojiang/miniconda3/etc/profile.d/conda.sh
conda activate xbd

# ensure TensorBoard log dir exists
TB_DIR="${SLURM_SUBMIT_DIR}/tb_logs"
mkdir -p "$TB_DIR"

# launch TensorBoard in the background
tensorboard --logdir "$TB_DIR" --host 0.0.0.0 --port 6006 & 
TB_PID=$!

# run distributed training
torchrun --nproc_per_node=2 -m src.train_controlnet_severity_ddp \
    --labels_dir    data/train/labels \
    --images_dir    data/train/images \
    --batch_size    2 \
    --lr            1e-4 \
    --epochs        10 \
    --ckpt_dir      checkpoints \
    --tensorboard_dir "$TB_DIR"

echo "===== JOB END $(date) ====="

# clean up TensorBoard
kill $TB_PID || true
