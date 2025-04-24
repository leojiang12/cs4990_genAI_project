# Final Project for CS4990 Special Topics for Undergraduate Students (Generative AI)

## Spring 2025

### Project structure (GPT-proposed)

xbd_scratch/ \
├── data/                   # your full 10 GB xBD dataset goes here \
├── notebooks/              # for exploratory EDA \
├── src/ \
│   ├── datasets.py         # PyTorch Dataset + transforms \
│   ├── models.py           # UNetGenerator & PatchDiscriminator \
│   ├── losses.py           # GAN loss, L1, perceptual (optional) \
│   ├── train.py            # training loop + CLI args \
│   └── utils.py            # logging, checkpointing, metrics \
├── configs/                # json/yaml hyperparameter files \
├── scripts/ \
│   └── sbatch_train.sh     # HPC submission script \
└── README.md \

### Create symlinks to point empty data directories to true directories

ln -s .../TRAIN_PATH         data/train     # Training dataset (tier 1 + tier 3) \
ln -s .../TEST_PATH         data/test       # Publically available test dataset \
ln -s .../TIER3_PATH         data/tier3     # Harder train cases \
ln -s .../HOLD_PATH         data/hold       # Judge's test set \

### Set-up conda environment:

`conda create -n xbd_scratch python=3.9 -y`\
`conda activate xbd_scratch`

To install dependencies, run from project root:

`pip install -r requirements.txt`


### To run, run this command from project root:

python -m src.train \\\
  --data_root data/train \\\
  --max_samples 1 \\\
  --crop_size 512 \\\
  --batch_size 4 \\\
  --epochs 5 \\\
  --log_interval 1 \\\
  --log_dir runs/test \\\
  --ckpt_dir checkpoints/test \\\