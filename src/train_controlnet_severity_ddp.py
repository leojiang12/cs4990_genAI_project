#!/usr/bin/env python3
import os, re
import argparse
import logging
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

from src.datasets import XBDFullDataset


def setup_ddp():
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size, local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # ── 1) load & freeze all but ControlNet ─────────────────
    vae          = AutoencoderKL.from_pretrained(
                        "runwayml/stable-diffusion-v1-5", subfolder="vae"
                   ).to(device)
    tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained(
                        "openai/clip-vit-large-patch14"
                   ).to(device)
    unet         = UNet2DConditionModel.from_pretrained(
                        "runwayml/stable-diffusion-v1-5", subfolder="unet"
                   ).to(device)
    controlnet   = ControlNetModel.from_pretrained(
                        "lllyasviel/sd-controlnet-depth"
                   ).to(device)
    scheduler    = DDPMScheduler.from_pretrained(
                        "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
                   )

    for p in vae.parameters():           p.requires_grad = False
    for p in unet.parameters():          p.requires_grad = False
    for p in text_encoder.parameters():  p.requires_grad = False
    for p in controlnet.parameters():    p.requires_grad = True

    controlnet = DDP(controlnet, device_ids=[local_rank], output_device=local_rank)

    # ── 2) build concatenated dataset ──────────────────────
    roots = [r.strip() for r in args.data_roots.split(",")]
    parts = []
    for root in roots:
        parts.append(XBDFullDataset(
            labels_root = os.path.join(root, "labels"),
            images_root = os.path.join(root, "images"),
            crop_size   = args.crop_size,
            max_samples = args.max_samples,
            annotate    = False,
        ))
    dataset = ConcatDataset(parts)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                         num_workers=4, pin_memory=True)

    if is_main(rank):
        logging.info(f"Total samples: {len(dataset)}")
        # TensorBoard
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.tensorboard_dir)
        logging.info(f"TensorBoard → {args.tensorboard_dir}")
    else:
        writer = None

    # ── 3) optimizer & optional resume ─────────────────────
    start_epoch = 1
    optimizer   = AdamW(controlnet.parameters(), lr=args.lr)

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        # new-format resume?
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            controlnet.module.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optim_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            if is_main(rank):
                logging.info(f"Resumed from {args.resume} at epoch {start_epoch}")
        else:
            # old-format (raw state_dict)
            controlnet.module.load_state_dict(ckpt)
            # try to parse epoch from filename: controlnet_epoch{N}.pth
            m = re.search(r"epoch(\d+)\.pth$", args.resume)
            if m:
                start_epoch = int(m.group(1)) + 1
                if is_main(rank):
                    logging.info(f"Loaded legacy checkpoint, starting at epoch {start_epoch}")
            else:
                if is_main(rank):
                    logging.info(f"Loaded legacy checkpoint, but couldn't infer epoch—starting at 1")

    # ── 4) train loop ─────────────────────────────────────
    total_steps = 0
    for epoch in range(start_epoch, args.epochs + 1):
        sampler.set_epoch(epoch)
        loop = tqdm(loader, desc=f"Epoch [{epoch}/{args.epochs}]", disable=rank!=0)
        for batch in loop:
            pre      = batch["pre"].to(device)
            post     = batch["post"].to(device)
            mask     = batch["mask"].to(device)           # [B,1,H,W]
            B,_,H,W  = mask.shape
            control  = mask.expand(B,3,H,W)               # [B,3,H,W]

            tokens      = tokenizer(["photo"]*B, return_tensors="pt",
                                    padding="max_length", truncation=True,
                                    max_length=tokenizer.model_max_length).to(device)
            text_embeds = text_encoder(**tokens).last_hidden_state

            with torch.no_grad():
                latents = vae.encode(post).latent_dist.sample() * vae.config.scaling_factor
            noise     = torch.randn_like(latents)
            tsteps    = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device)
            noisy     = scheduler.add_noise(latents, noise, tsteps)

            ctrl_out  = controlnet(noisy, timestep=tsteps,
                                   encoder_hidden_states=text_embeds,
                                   controlnet_cond=control)
            down, mid = ctrl_out.down_block_res_samples, ctrl_out.mid_block_res_sample

            unet_out  = unet(noisy, timestep=tsteps,
                             encoder_hidden_states=text_embeds,
                             down_block_additional_residuals=down,
                             mid_block_additional_residual=mid)
            pred      = unet_out.sample

            loss = torch.nn.functional.mse_loss(pred, noise)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            total_steps += 1
            if is_main(rank):
                writer.add_scalar("train/loss", loss.item(), total_steps)
                loop.set_postfix(loss=f"{loss.item():.4f}")

        # ── save checkpoint ──────────────────────────────
        if is_main(rank):
            out_ckpt = os.path.join(args.ckpt_dir, f"controlnet_epoch{epoch}.pth")
            torch.save({
                "epoch":            epoch,
                "model_state_dict": controlnet.module.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
            }, out_ckpt)
            logging.info(f"Checkpoint → {out_ckpt}")

    if writer:
        writer.close()
    if is_main(rank):
        logging.info("=== Training Complete ===")
    cleanup_ddp()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_roots",      type=str, required=True,
                   help="CSV of roots, e.g. data/train,data/tier3,data/test")
    p.add_argument("--crop_size",       type=int, default=512)
    p.add_argument("--max_samples",     type=int, default=None)
    p.add_argument("--batch_size",      type=int, default=4)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--epochs",          type=int, default=10)
    p.add_argument("--ckpt_dir",        type=str, default="checkpoints")
    p.add_argument("--tensorboard_dir", type=str, default="tb_logs")
    p.add_argument("--resume",          type=str, default=None,
                   help="path to a resume checkpoint (dict with model+optim+epoch)")
    args = p.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    main(args)
