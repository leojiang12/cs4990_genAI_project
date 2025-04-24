import os
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

from src.datasets import XBDPairDataset
from src.models   import UNetGenerator, PatchDiscriminator
from src.losses   import adversarial_loss, l1_loss

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{datetime.now()}] Using device: {device}")

    # — build dataset & loader —
    ds = XBDPairDataset(
        labels_dir  = f"{cfg.data_root}/labels",
        images_dir  = f"{cfg.data_root}/images",
        crop_size   = cfg.crop_size,
        max_samples = cfg.max_samples,
        annotate    = False,
    )
    print(f"[{datetime.now()}] Found {len(ds)} samples in dataset")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    # — determine input channels —
    gen_in_ch  = 3 + (1 if cfg.use_mask else 0)
    disc_in_ch = gen_in_ch + 3

    # — model, optimizers, logger —
    gen  = UNetGenerator(in_ch=gen_in_ch, out_ch=3).to(device)
    disc = PatchDiscriminator(in_ch=disc_in_ch).to(device)
    opt_g = Adam(gen.parameters(),  lr=cfg.lr, betas=(0.5,0.999))
    opt_d = Adam(disc.parameters(), lr=cfg.lr, betas=(0.5,0.999))
    tb    = SummaryWriter(log_dir=cfg.log_dir)
    print(f"[{datetime.now()}] Models initialized. Logging to `{cfg.log_dir}`")

    for epoch in range(1, cfg.epochs+1):
        print(f"\n[{datetime.now()}] Starting epoch {epoch}/{cfg.epochs}")
        epoch_iter = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}", unit="batch")
        for i, batch in epoch_iter:
            pre  = batch["pre"].to(device)
            post = batch["post"].to(device)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(device)

            # build input
            inp = torch.cat([pre, mask], dim=1) if cfg.use_mask else pre

            # 1) Discriminator step
            fake        = gen(inp)
            real_logits = disc(inp, post)
            fake_logits = disc(inp, fake.detach())
            d_loss = 0.5 * (adversarial_loss(real_logits, torch.ones_like(real_logits)) +
                            adversarial_loss(fake_logits, torch.zeros_like(fake_logits)))
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            # 2) Generator step
            g_adv  = adversarial_loss(disc(inp, fake), torch.ones_like(real_logits))
            g_l1   = l1_loss(fake, post) * cfg.l1_weight
            g_loss = g_adv + g_l1
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()

            # TensorBoard scalars
            step = (epoch-1)*len(loader) + i
            tb.add_scalar("D_loss", d_loss.item(), step)
            tb.add_scalar("G_loss", g_loss.item(), step)

            # Terminal logging every log_interval
            if (i + 1) % cfg.log_interval == 0 or i == len(loader)-1:
                msg = (f"[{datetime.now()}] Epoch {epoch}/{cfg.epochs} "
                       f"Batch {i+1}/{len(loader)} | "
                       f"D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")
                print(msg)

        # End-of-epoch checkpoint & sample
        if epoch % cfg.checkpoint_interval == 0:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"gen_epoch{epoch}.pth")
            os.makedirs(cfg.ckpt_dir, exist_ok=True)
            torch.save(gen.state_dict(), ckpt_path)
            print(f"[{datetime.now()}] Saved checkpoint: {ckpt_path}")

        # Log sample images to TensorBoard
        with torch.no_grad():
            n = min(4, pre.size(0))
            sample_inp  = inp[:n]
            sample_post = post[:n]
            sample_fake = gen(sample_inp)
            display = torch.cat([
                sample_inp[:, :3, ...], sample_post, sample_fake
            ], dim=0)
            grid = torchvision.utils.make_grid((display + 1) / 2, nrow=n)
            tb.add_image("samples", grid, epoch)

    tb.close()
    print(f"[{datetime.now()}] Training complete.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   type=str, required=True,
                   help="root of: labels/  images/{pre_disaster,post_disaster}/")
    p.add_argument("--max_samples", type=int,   default=None)
    p.add_argument("--crop_size",   type=int,   default=512)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--l1_weight",   type=float, default=100.0)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--log_dir",     type=str,   default="runs")
    p.add_argument("--ckpt_dir",    type=str,   default="checkpoints")
    p.add_argument("--log_interval",type=int,   default=100,
                   help="batches between terminal logs")
    p.add_argument("--checkpoint_interval",type=int,   default=10,
                   help="batches between checkpoints")
    p.add_argument("--use_mask",    action="store_true",
                   help="concat polygon mask as extra channel")
    cfg = p.parse_args()
    train(cfg)
