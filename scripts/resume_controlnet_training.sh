#!/usr/bin/env bash
# scripts/resume_controlnet_training.sh

set -euo pipefail

# where your DDP entry‐point lives:
TRAIN_SCRIPT=src/train_controlnet_severity_ddp.py

# same args you used before:
DATA_ROOTS="data/train,data/tier3,data/test"
CKPT_DIR="checkpoints"
TB_DIR="tb_logs"
BATCH_SIZE=4
LR=1e-4
EPOCHS=10

# find the latest checkpoint in $CKPT_DIR
LAST=$(ls -1 ${CKPT_DIR}/controlnet_epoch*.pth 2>/dev/null \
       | sort -V | tail -n1)

if [[ -z "$LAST" ]]; then
  echo "No existing checkpoint found in $CKPT_DIR. Starting fresh."
  RESUME_ARG=""
else
  echo "Resuming from checkpoint: $LAST"
  RESUME_ARG="--resume_from_checkpoint $LAST"
fi

torchrun --nproc_per_node=2 -m $TRAIN_SCRIPT \
  --data_roots       $DATA_ROOTS \
  --batch_size       $BATCH_SIZE \
  --lr               $LR \
  --epochs           $EPOCHS \
  --ckpt_dir         $CKPT_DIR \
  --tensorboard_dir  $TB_DIR \
  $RESUME_ARG
