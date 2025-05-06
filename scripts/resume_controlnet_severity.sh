#!/bin/bash
#SBATCH --job-name=controln-resume
#SBATCH --partition=gpu          # still on GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4        # for DataLoader workers
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=8G         # 8 GB × 8 CPUs = 64 GB total
#SBATCH --time=21-00:00:00
#SBATCH --output=logs/sd_control_train_resume_%j.log
#SBATCH --error=logs/sd_control_train_resume_%j.err

echo "===== JOB START $(date) ====="

# 1) Activate conda
source /data03/home/leojiang/miniconda3/etc/profile.d/conda.sh
conda activate xbd

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
python -c "import torch; print('torch.cuda.device_count()=', torch.cuda.device_count())"

# 2) Find latest checkpoint
CKPT_DIR="${SLURM_SUBMIT_DIR}/checkpoints"
LATEST=$(ls ${CKPT_DIR}/controlnet_epoch*.pth 2>/dev/null | sort | tail -n1)
if [[ -n "$LATEST" ]]; then
    echo "Resuming from $LATEST"
    RESUME_ARG="--resume $LATEST"
else
    echo "No checkpoint found → starting from scratch"
    RESUME_ARG=""
fi

# 3) Make TensorBoard dir
TB_DIR="${SLURM_SUBMIT_DIR}/tb_logs"
mkdir -p "$TB_DIR"

# 4) Launch distributed training
torchrun --nproc_per_node=2 -m src.train_controlnet_severity_ddp \
    --data_roots       data/train,data/tier3,data/test \
    --crop_size        512 \
    --batch_size       2 \
    --lr               1e-4 \
    --epochs           15 \
    --ckpt_dir         checkpoints \
    --tensorboard_dir  "$TB_DIR" \
    $RESUME_ARG

echo "===== JOB END $(date) ====="
