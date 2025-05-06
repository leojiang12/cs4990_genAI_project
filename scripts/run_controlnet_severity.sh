#!/bin/bash
#SBATCH --job-name=controln
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -t 21-00:00:00
#SBATCH --output=logs/sd_control_train_%j.log
#SBATCH --error=logs/sd_control_train_%j.err

RUN_NAME=${1:-"controlnet_${SLURM_JOB_ID}"}

echo "===== JOB START $(date) ====="

# 1) Conda
source /data03/home/leojiang/miniconda3/etc/profile.d/conda.sh
conda activate xbd

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
python -c "import torch; print('torch.cuda.device_count()=', torch.cuda.device_count())"

# 2) TensorBoard logs
TB_DIR="${SLURM_SUBMIT_DIR}/tb_logs/${RUN_NAME}"
mkdir -p "$TB_DIR"

# 3) Checkpoint directory
CKPT_DIR="${SLURM_SUBMIT_DIR}/checkpoints"
mkdir -p "$CKPT_DIR"

# 4) Train
torchrun --nproc_per_node=2 -m src.train_controlnet_severity_ddp \
  --data_roots data/train,data/tier3 \
  --val_root   data/test \
  --seed       42 \
  --crop_size 512 \
  --batch_size 2 \
  --lr 1e-4 \
  --epochs 15 \
  --ckpt_dir checkpoints \
  --ckpt_dir  "$CKPT_DIR" \
  --tensorboard_dir "$TB_DIR"

echo "===== TRAINING COMPLETE $(date) ====="

# 5) Inference + visualization
#   We’ll pull the last checkpoint and run the “full” sweep script:
CKPT="${CKPT_DIR}/${RUN_NAME}_epoch15.pth"
OUT_IMG="logs/posttrain_inference_${SLURM_JOB_ID}.png"

python scripts/infer_and_visualize.py \
  --ckpt "$CKPT" \
  --data_root data/hold \
  --max_samples 4 \
  --out "$OUT_IMG"

echo "Saved post‑training visualization to $OUT_IMG"
echo "===== JOB END $(date) ====="
