#!/bin/bash
#SBATCH --job-name=controlnet-sd
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -t 21-00:00:00
#SBATCH --output=logs/sd_control_train_%j.log
#SBATCH --error=logs/sd_control_train_%j.err

RUN_NAME=${1:-"controlnet_${SLURM_JOB_ID}"}

echo "===== JOB START $(date) (run=${RUN_NAME}) ====="

# 1) environment
source /data03/home/leojiang/miniconda3/etc/profile.d/conda.sh
conda activate xbd

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python - <<<'import torch;print("GPUs:",torch.cuda.device_count())'

# 2) create dirs
TB_DIR="${SLURM_SUBMIT_DIR}/tb_logs/${RUN_NAME}"
CKPT_DIR="${SLURM_SUBMIT_DIR}/checkpoints"
mkdir -p "$TB_DIR" "$CKPT_DIR"

# 3) launch
torchrun --nproc_per_node=2 -m src.train_controlnet_severity_ddp \
  --data_roots data/train,data/tier3 \
  --val_root   data/test \
  --seed       42 \
  --run_name  "$RUN_NAME" \
  --crop_size 512 \
  --batch_size 2 \
  --lr 1e-4 \
  --epochs 15 \
  --ckpt_dir "$CKPT_DIR" \
  --tensorboard_dir "$TB_DIR" \
  ${@:2}   # pass any extra flags

echo "===== TRAINING COMPLETE $(date) ====="

# 4) inference sweep
LAST_CKPT=$(ls -1 "${CKPT_DIR}/${RUN_NAME}"*epoch15.pth | tail -n1)
if [ -z "$LAST_CKPT" ]; then
  echo "Checkpoint not found!"
  exit 1
fi

OUT_IMG="logs/posttrain_inference_${SLURM_JOB_ID}.png"
python scripts/infer_and_visualize.py \
  --ckpt "$LAST_CKPT" \
  --data_root data/hold \
  --max_samples 4 \
  --out "$OUT_IMG"

echo "Saved post‑training visualization → $OUT_IMG"
echo "===== JOB END $(date) ====="
