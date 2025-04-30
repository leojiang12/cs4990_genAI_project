#!/usr/bin/env python3
"""
Standalone validation script for SetTransformerRegressor without distributed setup.
"""
import os
import glob
import argparse
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from src.data.custom_cg_dataset import CustomCGDataset
from src.models.set_transformer import SetTransformerRegressor
from src.utils.training_utils import init_weights, run_evaluation
from src.utils.config import FAULTY_SAMPLES_FILE


def main():
    parser = argparse.ArgumentParser(
        description="Validate a trained SetTransformerRegressor on a dataset (single-GPU/CPU)."
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model checkpoint (.pth file)")
    parser.add_argument("--d_model", type=int, default=32,
                        help="Dimension of the model's embeddings")
    parser.add_argument("--n_heads", type=int, default=2,
                        help="Number of attention heads in the SetTransformer")
    parser.add_argument("--num_encoder_layers", type=int, default=4,
                        help="Number of encoder layers in the SetTransformer")
    parser.add_argument("--dim_feedforward", type=int, default=128,
                        help="Hidden dimension of the feedforward networks")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument("--output_dim", type=int, default=1,
                        help="Output dimension of the regressor head")
    parser.add_argument("--num_seeds", type=int, default=32,
                        help="Number of seed vectors for the PMA layer")
    parser.add_argument("--mlp_embed", action="store_true",
                        help="Enable MLP embedding before the PMA layer")
    parser.add_argument("--mlp_embed_h", type=int, default=32,
                        help="Hidden size for the MLP embedder")

    parser.add_argument("--data", type=str, default="all",
                        help="Comma-separated subdirs under ./data/data_exact_pt or 'all'")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Random seed for dataset split")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Disable input normalization in the collate function")
    parser.add_argument("--no_sci", action="store_true",
                        help="Disable scientific mode for the dataset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize_inputs = not args.no_normalize
    scientific = not args.no_sci

    # Build dataset directories
    data_root = "./data/data_exact_pt"
    raw = args.data.strip()
    if raw.lower() == "all":
        dirs = [d for d in glob.glob(os.path.join(data_root, "*"))
                if os.path.isdir(d) and (
                    any(suffix in os.path.basename(d) for suffix in ["_20k_pt", "_20k_inv_pt"])  )]
    else:
        names = [n.strip() for n in raw.split(",") if n.strip()]
        dirs = [os.path.join(data_root, name) for name in names]

    dataset = CustomCGDataset(dirs, allowed_dirs=None,
                              scientific=scientific,
                              name="validation_dataset")
    # Exclude faulty samples
    with open(FAULTY_SAMPLES_FILE, "r") as f:
        excluded = [line.strip() for line in f if line.strip()]
    dataset.append_excluded_files(excluded)

    # Split into train/val
    total = len(dataset)
    val_size = int(0.2 * total)
    train_size = total - val_size
    generator = torch.Generator().manual_seed(args.split_seed)
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # Build model
    input_dim = dataset[0]["eig_val"].shape[-1]
    model = SetTransformerRegressor(
        input_dim=input_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        output_dim=args.output_dim,
        num_seeds=args.num_seeds,
        mlp_embed=args.mlp_embed,
        mlp_embed_h=args.mlp_embed_h
    )
    model.apply(init_weights)

    # Load checkpoint
    state = torch.load(args.model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Evaluate
    run_evaluation(model, val_dataset, 0, normalize_inputs, writer=None)

if __name__ == "__main__":
    main()
