# Final Project for CS4990 Special Topics for Undergraduate Students (Generative AI)

## Spring 2025

### Project structure (GPT-proposed)

xbd_scratch/ <br\>
├── data/                   # your full 10 GB xBD dataset goes here <br\>
├── notebooks/              # for exploratory EDA <br\>
├── src/ <br\>
│   ├── datasets.py         # PyTorch Dataset + transforms <br\>
│   ├── models.py           # UNetGenerator & PatchDiscriminator <br\>
│   ├── losses.py           # GAN loss, L1, perceptual (optional) <br\>
│   ├── train.py            # training loop + CLI args <br\>
│   └── utils.py            # logging, checkpointing, metrics <br\>
├── configs/                # json/yaml hyperparameter files <br\>
├── scripts/ <br\>
│   └── sbatch_train.sh     # HPC submission script <br\>
└── README.md <br\>

### Create symlinks to point empty data directories to true directories

ln -s .../TRAIN_PATH         data/train     # Training dataset (tier 1 + tier 3) <br\>
ln -s .../TEST_PATH         data/test       # Publically available test dataset <br\>
ln -s .../TIER3_PATH         data/tier3     # Harder train cases <br\>
ln -s .../HOLD_PATH         data/hold       # Judge's test set <br\>

### To run, run this command from project root:

python -m src.train \\
  --data_root data/train \\
  --max_samples 1 \\
  --crop_size 512 \\
  --batch_size 4 \\
  --epochs 5 \\
  --log_interval 1 \\ 
  --log_dir runs/test \\
  --ckpt_dir checkpoints/test \\