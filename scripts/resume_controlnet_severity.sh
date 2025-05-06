#!/bin/bash
#SBATCH --job-name=controln-resume
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_resume_%j.log
#SBATCH --error=logs/train_resume_%j.err

echo "===== JOB START $(date) ====="

# 1) Activate conda (keeps SLURM’s CUDA_VISIBLE_DEVICES)
source /data03/home/leojiang/miniconda3/etc/profile.d/conda.sh
conda activate xbd

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
python -c "import torch; print('torch.cuda.device_count()=', torch.cuda.device_count())"

# 2) find latest checkpoint (or leave empty to start from scratch)
CKPT_DIR="${SLURM_SUBMIT_DIR}/checkpoints"
LATEST="$(ls ${CKPT_DIR}/controlnet_epoch*.pth 2>/dev/null | sort | tail -n1 || true)"
if [ -n "$LATEST" ]; then
    echo "Resuming from $LATEST"
    RESUME_ARG="--resume $LATEST"
else
    echo "No checkpoint found in $CKPT_DIR → starting from scratch"
    RESUME_ARG=""
fi

# 3) TensorBoard logs
TB_DIR="${SLURM_SUBMIT_DIR}/tb_logs"
mkdir -p "$TB_DIR"

# 4) launch distributed training
torchrun --nproc_per_node=2 -m src.train_controlnet_severity_ddp \
    --data_roots      data/train,data/tier3,data/test \
    --crop_size       512 \
    --batch_size      2 \
    --lr              1e-4 \
    --epochs          15 \
    --ckpt_dir        checkpoints \
    --tensorboard_dir tb_logs \
    $RESUME_ARG

echo "===== JOB END   $(date) ====="
