#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate xbd_scratch

python src/train.py \
  --data_root /scratch/user/xb_pairs \
  --crop_size 512 \
  --batch_size 4 \
  --epochs 100 \
  --log_dir /scratch/user/logs \
  --ckpt_dir /scratch/user/ckpts
