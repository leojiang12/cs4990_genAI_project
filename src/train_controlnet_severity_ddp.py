#!/usr/bin/env python3
import os
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

from src.datasets import XBDFullDataset  # <-- switched to recursive version


def setup_ddp(args):
    args.rank       = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank
    )


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    setup_ddp(args)
    device = torch.device(f"cuda:{args.local_rank}")

    # ── 1) Load & freeze models ────────────────────────
    vae          = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
    tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    unet         = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(device)
    controlnet   = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth").to(device)
    scheduler    = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    for p in vae.parameters():           p.requires_grad = False
    for p in unet.parameters():          p.requires_grad = False
    for p in text_encoder.parameters():  p.requires_grad = False
    for p in controlnet.parameters():    p.requires_grad = True

    controlnet = DDP(controlnet, device_ids=[args.local_rank], output_device=args.local_rank)

    # ── 2) Dataset & Dataloader ────────────────────────
    roots  = [r.strip() for r in args.data_roots.split(",") if r.strip()]
    parts  = []
    for root in roots:
        logging.info(f"Building XBDFullDataset for {root}")
        parts.append(XBDFullDataset(
            labels_root = os.path.join(root, "labels"),
            images_root = os.path.join(root, "images"),
            crop_size   = args.crop_size,
            max_samples = args.max_samples,
            annotate    = False,
        ))
        logging.info(f"  → {len(parts[-1])} samples in {root}")
    ds = ConcatDataset(parts)
    total = len(ds)
    if is_main_process():
        logging.info(f"Total concatenated samples: {total}")

    sampler = DistributedSampler(ds, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    loader  = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                         num_workers=4, pin_memory=True)

    # ── 3) TensorBoard ──────────────────────────────────
    writer = None
    if is_main_process() and args.tensorboard_dir:
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.tensorboard_dir)
        logging.info(f"TensorBoard logs → {args.tensorboard_dir}")

    # ── 4) Optimizer ────────────────────────────────────
    optimizer = AdamW(controlnet.parameters(), lr=args.lr)

    # ── 5) Training Loop ────────────────────────────────
    total_steps = 0
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        loop = tqdm(loader, desc=f"Epoch [{epoch}/{args.epochs}]", disable=not is_main_process())
        for batch in loop:
            pre      = batch["pre"].to(device)
            post     = batch["post"].to(device)
            severity = batch["severity"].to(device).unsqueeze(-1)

            B, _, H, W = pre.shape
            control_map = severity.view(B,1,1,1).expand(B,3,H,W)

            tokens      = tokenizer(["photo"]*B, return_tensors="pt", padding="max_length",
                                    truncation=True, max_length=tokenizer.model_max_length).to(device)
            text_embeds = text_encoder(**tokens).last_hidden_state

            with torch.no_grad():
                latents = vae.encode(post).latent_dist.sample() * vae.config.scaling_factor
            noise     = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device)
            noisy     = scheduler.add_noise(latents, noise, timesteps)

            ctrl_out    = controlnet(noisy, timestep=timesteps,
                                     encoder_hidden_states=text_embeds,
                                     controlnet_cond=control_map)
            down_samps  = ctrl_out.down_block_res_samples
            mid_samp    = ctrl_out.mid_block_res_sample

            unet_out    = unet(noisy, timestep=timesteps,
                               encoder_hidden_states=text_embeds,
                               down_block_additional_residuals=down_samps,
                               mid_block_additional_residual=mid_samp)
            pred        = unet_out.sample

            loss = torch.nn.functional.mse_loss(pred, noise)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            total_steps += 1
            loop.set_postfix(loss=f"{loss:.4f}")
            if is_main_process() and writer:
                writer.add_scalar("train/loss", loss.item(), total_steps)

        if is_main_process():
            ckpt = os.path.join(args.ckpt_dir, f"controlnet_epoch{epoch}.pth")
            torch.save(controlnet.module.state_dict(), ckpt)
            logging.info(f"Saved checkpoint → {ckpt}")

    if writer:
        writer.close()
    if is_main_process():
        logging.info("=== Training Complete ===")
    cleanup_ddp()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_roots",      type=str,   required=True,
                   help="Comma-separated roots: data/train,data/tier3,data/test")
    p.add_argument("--crop_size",       type=int,   default=512)
    p.add_argument("--max_samples",     type=int,   default=None)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--epochs",          type=int,   default=10)
    p.add_argument("--ckpt_dir",        type=str,   default="checkpoints")
    p.add_argument("--tensorboard_dir", type=str,   default="tb_logs")
    args = p.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    main(args)
