# src/evaluate.py
import os, argparse
import torch
import matplotlib
# Use a non‐interactive backend so we can save files even on headless machines
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.datasets import XBDPairDataset
from src.models   import UNetGenerator
from src.losses   import l1_loss

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load generator
    gen = UNetGenerator().to(device)
    gen.load_state_dict(torch.load(args.checkpoint, map_location=device))
    gen.eval()

    # 2) load dataset
    ds = XBDPairDataset(
        labels_dir  = args.labels_dir,
        images_dir  = args.images_root,
        crop_size   = args.crop_size,
        max_samples = args.max_samples,
    )

    # 3) pick one example
    item = ds[args.sample_idx]
    pre, post = item["pre"].unsqueeze(0).to(device), item["post"].unsqueeze(0).to(device)

    # 4) forward + compute L1
    with torch.no_grad():
        fake = gen(pre)
        loss = l1_loss(fake, post).item()

    # 5) plot & save
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(((pre[0].cpu().permute(1,2,0).numpy()) + 1)/2)
    axes[0].set_title("Pre")
    axes[1].imshow(((post[0].cpu().permute(1,2,0).numpy())+1)/2)
    axes[1].set_title("Real Post")
    axes[2].imshow(((fake[0].cpu().permute(1,2,0).numpy())+1)/2)
    axes[2].set_title(f"Gen (L1={loss:.4f})")

    # Determine output path
    out_path = args.output or f"eval_sample_{args.sample_idx}.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"Saved evaluation figure to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels_dir",  required=True,
                   help="where the JSON labels live")
    p.add_argument("--images_root", required=True,
                   help="where the pre/post .pngs live")
    p.add_argument("--checkpoint",  required=True,
                   help="path to the generator .pth file")
    p.add_argument("--sample_idx",  type=int, default=0,
                   help="which example to visualize")
    p.add_argument("--crop_size",   type=int, default=512)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--output",      type=str, default=None,
                   help="where to save the output image (defaults to eval_sample_<idx>.png)")
    args = p.parse_args()
    evaluate(args)
