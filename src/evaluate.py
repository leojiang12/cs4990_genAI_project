# src/evaluate.py
import torch, argparse
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

    # 2) load a tiny JSON‐based dataset
    ds = XBDPairDataset(
        labels_dir  = args.labels_dir,
        images_root = args.images_root,
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

    # 5) display
    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    for ax in axes: ax.axis("off")
    axes[0].imshow(((pre[0].cpu().permute(1,2,0).numpy()) + 1)/2);   axes[0].set_title("Pre")
    axes[1].imshow(((post[0].cpu().permute(1,2,0).numpy())+1)/2);   axes[1].set_title("Real Post")
    axes[2].imshow(((fake[0].cpu().permute(1,2,0).numpy()) +1)/2);   axes[2].set_title(f"Gen (L1={loss:.4f})")
    plt.show()
