# src/train_ddp.py

import os
import argparse
import socket
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.datasets import XBDPairDataset
from src.models   import UNetGenerator, PatchDiscriminator
from src.losses   import adversarial_loss, l1_loss

torch.autograd.set_detect_anomaly(True)


def setup_ddp(args):
    """Initialize torch.distributed."""
    # Usually launched via torchrun which sets these env vars.
    args.rank       = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank
    )
    torch.cuda.set_device(args.local_rank)

def cleanup_ddp():
    dist.destroy_process_group()

def is_main_process():
    return dist.get_rank() == 0

def train_ddp(args):
    setup_ddp(args)
    device = torch.device(f"cuda:{args.local_rank}")
    
    if is_main_process():
        print(f"[{datetime.now()}] (rank {args.rank}/{args.world_size}) Using device: {device}")

    # — build dataset & distributed sampler —
    ds = XBDPairDataset(
        labels_dir  = os.path.join(args.data_root, "labels"),
        images_dir  = os.path.join(args.data_root, "images"),
        crop_size   = args.crop_size,
        max_samples = args.max_samples,
        annotate    = False
    )
    sampler = DistributedSampler(ds, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    loader  = DataLoader(ds,
                         batch_size=args.batch_size,
                         sampler=sampler,
                         num_workers=4,
                         pin_memory=True)
    
    if is_main_process():
        print(f"[{datetime.now()}] Dataset size: {len(ds)}, batches per epoch: {len(loader)}")
    
    # — determine channels & build models —
    gen_in_ch  = 3 + (1 if args.use_mask else 0)
    disc_in_ch = gen_in_ch + 3

    gen  = UNetGenerator(in_ch=gen_in_ch, out_ch=3).to(device)
    disc = PatchDiscriminator(in_ch=disc_in_ch).to(device)

    gen  = DDP(gen,  device_ids=[args.local_rank], output_device=args.local_rank)
    disc = DDP(disc, device_ids=[args.local_rank], output_device=args.local_rank)

    opt_g = Adam(gen.parameters(),  lr=args.lr, betas=(0.5,0.999))
    opt_d = Adam(disc.parameters(), lr=args.lr, betas=(0.5,0.999))

    # — only main process logs to TensorBoard —
    tb = None
    if is_main_process():
        tb = SummaryWriter(log_dir=args.log_dir)
        print(f"[{datetime.now()}] Logging to {args.log_dir}")

    # training loop
    for epoch in range(1, args.epochs+1):
        sampler.set_epoch(epoch)
        if is_main_process():
            print(f"\n[{datetime.now()}] Starting epoch {epoch}/{args.epochs}")
        epoch_iter = tqdm(enumerate(loader), total=len(loader),
                          desc=f"Rank {args.rank} Epoch {epoch}", disable=not is_main_process())
        
        for i, batch in epoch_iter:
            pre  = batch["pre"].to(device)
            post = batch["post"].to(device)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(device)

            inp = torch.cat([pre, mask], dim=1) if args.use_mask else pre

            # 1) discriminator step
            fake        = gen(inp)
            real_logits = disc(inp, post)
            fake_logits = disc(inp, fake.detach())
            d_loss = 0.5 * (
                adversarial_loss(real_logits, torch.ones_like(real_logits)) +
                adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
            )
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            # 2) generator step
            g_adv  = adversarial_loss(disc(inp, fake), torch.ones_like(real_logits))
            g_l1   = l1_loss(fake, post) * args.l1_weight
            g_loss = g_adv + g_l1
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()

            # log scalars on main
            if is_main_process():
                step = (epoch-1)*len(loader) + i
                tb.add_scalar("D_loss", d_loss.item(), step)
                tb.add_scalar("G_loss", g_loss.item(), step)
                if (i+1) % args.log_interval == 0 or i == len(loader)-1:
                    print(f"[{datetime.now()}] Epoch {epoch}/{args.epochs} "
                          f"Batch {i+1}/{len(loader)} | "
                          f"D_loss={d_loss:.4f} G_loss={g_loss:.4f}")

        # checkpoint on main process
        if is_main_process() and (epoch % args.checkpoint_interval == 0):
            os.makedirs(args.ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(args.ckpt_dir, f"gen_epoch{epoch}.pth")
            torch.save(gen.module.state_dict(), ckpt_path)
            print(f"[{datetime.now()}] Saved checkpoint {ckpt_path}")

        # sample to TensorBoard on main
        if is_main_process():
            with torch.no_grad():
                n = min(4, pre.size(0))
                sample_inp  = inp[:n]
                sample_post = post[:n]
                sample_fake = gen(sample_inp)
                disp = torch.cat([sample_inp[:, :3], sample_post, sample_fake], dim=0)
                grid = torchvision.utils.make_grid((disp + 1)/2, nrow=n)
                tb.add_image("samples", grid, epoch)

    if is_main_process():
        tb.close()
        print(f"[{datetime.now()}] Training complete.")

    cleanup_ddp()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",            type=str, required=True)
    p.add_argument("--max_samples",          type=int, default=None)
    p.add_argument("--crop_size",            type=int, default=512)
    p.add_argument("--batch_size",           type=int, default=8)
    p.add_argument("--lr",                   type=float, default=2e-4)
    p.add_argument("--l1_weight",            type=float, default=100.0)
    p.add_argument("--epochs",               type=int, default=50)
    p.add_argument("--log_dir",              type=str,   default="runs")
    p.add_argument("--ckpt_dir",             type=str,   default="checkpoints")
    p.add_argument("--log_interval",         type=int,   default=100)
    p.add_argument("--checkpoint_interval",  type=int,   default=1)
    p.add_argument("--use_mask",             action="store_true")
    args = p.parse_args()

    # launch via torchrun: torchrun --nproc_per_node=4 src/train_ddp.py --data_root data/train ...
    train_ddp(args)
