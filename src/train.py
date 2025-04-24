import os
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from src.datasets import XBDPairDataset
from src.models   import UNetGenerator, PatchDiscriminator
from src.losses   import adversarial_loss, l1_loss

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # — build dataset & loader —
    ds = XBDPairDataset(
    labels_dir  = f"{cfg.data_root}/labels",
    images_dir  = f"{cfg.data_root}/images",
    crop_size   = 512,
    max_samples = 100,
    annotate    = False,            # set True to draw polygons on returned PILs
    )
    print(f"Training over {len(ds)} pre-post disaster image pairs.")
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    pre, post, mask = batch["pre"], batch["post"], batch["mask"]

    # — determine input channels for Generator & Discriminator —
    # base: 3 for RGB. +1 if we concat the mask.
    gen_in_ch  = 3 + (1 if cfg.use_mask else 0)
    disc_in_ch = gen_in_ch + 3  # input+real/fake concat

    gen  = UNetGenerator(in_ch=gen_in_ch, out_ch=3).to(device)
    disc = PatchDiscriminator(in_ch=disc_in_ch).to(device)

    opt_g = Adam(gen.parameters(),  lr=cfg.lr, betas=(0.5,0.999))
    opt_d = Adam(disc.parameters(), lr=cfg.lr, betas=(0.5,0.999))
    tb    = SummaryWriter(log_dir=cfg.log_dir)

    for epoch in range(cfg.epochs):
        for i, batch in enumerate(loader):
            # unpack
            pre  = batch["pre"].to(device)   # [B,3,H,W]
            post = batch["post"].to(device)  # [B,3,H,W]
            mask = batch.get("mask")         # None or [B,1,H,W]
            if mask is not None:
                mask = mask.to(device)

            # build generator input (RGB [+ mask])
            if cfg.use_mask:
                assert mask is not None, "use_mask=True but no mask in batch!"
                inp = torch.cat([pre, mask], dim=1)
            else:
                inp = pre

            # 1) Discriminator update (using two-arg forward)
            fake        = gen(inp)
            real_logits = disc(inp, post)
            fake_logits = disc(inp, fake.detach())

            d_loss     = 0.5 * (adversarial_loss(real_logits, torch.ones_like(real_logits)) +
                                adversarial_loss(fake_logits, torch.zeros_like(fake_logits)))

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # 2) Generator update
            g_adv = adversarial_loss(disc(inp, fake), torch.ones_like(real_logits))

            g_l1       = l1_loss(fake, post) * cfg.l1_weight
            g_loss     = g_adv + g_l1

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            # logging
            if i % cfg.log_interval == 0:
                step = epoch * len(loader) + i
                tb.add_scalar("D_loss", d_loss.item(), step)
                tb.add_scalar("G_loss", g_loss.item(), step)

        # end of epoch → checkpoint & sample
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        torch.save(gen.state_dict(),
                   os.path.join(cfg.ckpt_dir, f"gen_epoch{epoch}.pth"))

        with torch.no_grad():
            # sample first few in-batch examples
            n = min(4, pre.size(0))
            sample_inp  = inp[:n]
            sample_post = post[:n]
            sample_fake = gen(sample_inp)

            # display pre | real | gen
            display = torch.cat([
                sample_inp[:, :3, ...],  # drop mask channel if present
                sample_post,
                sample_fake
            ], dim=0)
            grid = torchvision.utils.make_grid((display + 1) / 2, nrow=n)
            tb.add_image("samples", grid, epoch)

    tb.close()


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
    p.add_argument("--log_interval",type=int,   default=100)
    p.add_argument("--use_mask",    action="store_true",
                   help="concat polygon mask as extra channel")
    cfg = p.parse_args()
    train(cfg)
