#!/bin/bash
#SBATCH --job-name=controln
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=cn01            # <- force allocation on cn01 only
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G                 # ← request 64 GB of RAM
#SBATCH --t 21-00:00:00
#SBATCH --output=logs/sd_control_train_%j.log
#SBATCH --error=logs/sd_control_train_%j.err

echo "===== JOB START $(date) ====="

# 1) Activate conda (preserves SLURM's CUDA_VISIBLE_DEVICES)
source /data03/home/leojiang/miniconda3/etc/profile.d/conda.sh
conda activate xbd

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
python -c "import torch; print('torch.cuda.device_count()=', torch.cuda.device_count())"

# 2) Prepare TensorBoard logdir (only rank0 will write to it)
TB_DIR="${SLURM_SUBMIT_DIR}/tb_logs"
mkdir -p "$TB_DIR"

# 3) Launch distributed training
torchrun --nproc_per_node=1 -m src.train_controlnet_severity_ddp \
  --data_roots data/train,data/tier3,data/test \
  --crop_size 512 --batch_size 2 --lr 1e-4 --epochs 50 \
  --ckpt_dir checkpoints --tensorboard_dir tb_logs

echo "===== JOB END $(date) ====="
