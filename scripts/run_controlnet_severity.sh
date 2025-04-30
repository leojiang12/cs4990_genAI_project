#!/bin/bash
#
#SBATCH --job-name=controlnet_sev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2            # torchrun --nproc_per_node=2
#SBATCH --gres=gpu:2                   # request 2 GPUs
#SBATCH --cpus-per-task=4
#SBATCH -t 3-00:00:00
#SBATCH --output=logs/train_%j.log     # %j = job ID


# make sure our logs directory exists
mkdir -p logs checkpoints tensorboard

# timestamped logfile
LOGFILE=logs/train_$(date +%Y%m%d_%H%M%S).log

echo "========== JOB START $(date) ==========" | tee -a ${LOGFILE}


# run training
conda run -n xbd torchrun --nproc_per_node=2 -m src.train_controlnet_severity_ddp \
    --labels_dir data/train/labels \
    --images_dir data/train/images \
    --batch_size 2 \
    --lr 1e-4 \
    --epochs 10 \
    --ckpt_dir checkpoints \
    --tensorboard_dir tensorboard \
  2>&1 | tee -a ${LOGFILE}

echo "========== JOB END   $(date) ==========" | tee -a ${LOGFILE}

# kill tensorboard on exit
kill ${TB_PID}
