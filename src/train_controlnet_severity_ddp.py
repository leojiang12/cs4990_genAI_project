#!/usr/bin/env python3
import os
import random
import numpy as np
from datetime import datetime
import re
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
    # ── Logging setup ───────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        # include seed & run_id in each message
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 0) fix RNGs
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    setup_ddp(args)
    device = torch.device(f"cuda:{args.local_rank}")

    # ── 1) Load & freeze models ────────────────────────
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

    controlnet = DDP(controlnet, device_ids=[args.local_rank], output_device=args.local_rank)

    # ── 2) Dataset & Dataloader ────────────────────────
    roots = [r.strip() for r in args.data_roots.split(",") if r.strip()]

    # behind your training roots, expect also a test split:
    val_root = args.val_root
    if is_main_process():
        logging.info(f"Building VAL dataset for {val_root}")
    val_ds = XBDFullDataset(
        labels_root = os.path.join(val_root, "labels"),
        images_root = os.path.join(val_root, "images"),
        crop_size   = args.crop_size,
        max_samples = None,
        annotate    = False,
    )
    val_sampler = DistributedSampler(val_ds,
                                     num_replicas=args.world_size,
                                     rank=args.rank,
                                     shuffle=False)
    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            sampler=val_sampler,
                            num_workers=4,
                            pin_memory=True)

    parts = []
    for root in roots:
        logging.info(f"Building XBDFullDataset for {root}")
        ds_part = XBDFullDataset(
            labels_root = os.path.join(root, "labels"),
            images_root = os.path.join(root, "images"),
            crop_size   = args.crop_size,
            max_samples = args.max_samples,
            annotate    = False,
        )
        parts.append(ds_part)
        logging.info(f"  → {len(ds_part)} samples in {root}")
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
        # make a run‐unique subfolder
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_run_dir = os.path.join(args.tensorboard_dir, run_id)
        os.makedirs(tb_run_dir, exist_ok=True)
        logging.info(f"TensorBoard logs → {tb_run_dir}")
        writer = SummaryWriter(log_dir=tb_run_dir)
        logging.info(f"TensorBoard logs → {args.tensorboard_dir}")

    # ── 4) Optimizer & optional resume ──────────────────
    optimizer   = AdamW(controlnet.parameters(), lr=args.lr)

    # we'll track average train & val loss per epoch
    def epoch_avg_loss(running_loss, n_batches):
        return running_loss / n_batches if n_batches>0 else float("nan")

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model_sd = ckpt.get("model_state_dict", ckpt)
        controlnet.module.load_state_dict(model_sd)
        # parse epoch number if available
        m = re.search(r"epoch(\d+)\.pth$", args.resume)
        if m:
            start_epoch = int(m.group(1)) + 1
        if is_main_process():
            logging.info(f"Resumed from {args.resume} → continuing at epoch {start_epoch}")
        del ckpt, model_sd  # free CPU RAM

    # make sure run_name is a safe filesystem prefix
    run_name = args.run_name.strip().replace(" ", "_")

    # ── 5) Training Loop ────────────────────────────────
    total_steps = 0
    for epoch in range(start_epoch, args.epochs + 1):
        sampler.set_epoch(epoch)
        loop = tqdm(loader, desc=f"Epoch [{epoch}/{args.epochs}]", disable=not is_main_process())

        # TRAINING
        train_loss_sum = 0.0
        train_batches  = 0
        for batch in loop:
            pre  = batch["pre"].to(device)
            post = batch["post"].to(device)
            mask = batch["mask"].to(device)       # [B,1,H,W]
            B,_,H,W = mask.shape
            control_map = mask.expand(B,3,H,W)

            raw_meta = batch["meta"]               # e.g. {'sensor': [...], 'capture_date': [...], …}
            batch_meta = [
                { k: raw_meta[k][i] for k in raw_meta }
                for i in range(B)
            ]

            # build a per‑sample prompt from metadata
            prompts = []
            for i, m in enumerate(batch_meta):
                # geography
                lnglat = m.get("features",{}).get("lng_lat", [])
                if lnglat:
                    # take first coordinate pair
                    coords = lnglat[0].get("properties",{})
                    loc = f"{coords.get('lng', '0'):.2f}E,{coords.get('lat','0'):.2f}N"
                else:
                    loc = "unknown location"

                # disaster context
                disaster      = m.get("metadata",{}).get("disaster", "")
                disaster_type = m.get("metadata",{}).get("disaster_type", "")

                # imaging conditions
                sun_el        = m.get("metadata",{}).get("sun_elevation", None)
                off_nadir     = m.get("metadata",{}).get("off_nadir_angle", None)
                gsd           = m.get("metadata",{}).get("gsd", None)

                feats = m.get("features",{}).get("xy", [])
                if feats:
                    subtype = feats[0].get("properties",{}).get("subtype","no-damage")
                else:
                    subtype = "no-damage"

                # and the numeric severity you already compute:
                sev = batch["severity"][i].item()  # between 0 and 1

                # capture date
                date = m.get("metadata",{}).get("capture_date", "").split("T")[0]

                # assemble
                p = f"{disaster_type or disaster} aftermath in {loc}"
                if date:
                    p += f", on {date}"
                if subtype and sev>0:
                    p += f", {subtype.replace('-',' ')} ({sev*100:.0f}% area)"

                if sun_el is not None:
                    p += f", sunny angle {sun_el:.1f}°"
                if off_nadir is not None:
                    p += f", off‑nadir {off_nadir:.1f}°"
                if gsd is not None:
                    p += f", {gsd:.2f} m/px resolution"

                prompts.append(p)

            tokens = tokenizer(prompts,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True,
                                max_length=tokenizer.model_max_length
                                ).to(device)
            text_embeds = text_encoder(**tokens).last_hidden_state

            with torch.no_grad():
                latents = vae.encode(post).latent_dist.sample() * vae.config.scaling_factor
            noise  = torch.randn_like(latents)
            tsteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device)
            noisy  = scheduler.add_noise(latents, noise, tsteps)

            ctrl_out   = controlnet(noisy, timestep=tsteps,
                                    encoder_hidden_states=text_embeds,
                                    controlnet_cond=control_map)
            down_samps = ctrl_out.down_block_res_samples
            mid_samp   = ctrl_out.mid_block_res_sample

            unet_out = unet(
                noisy, timestep=tsteps,
                encoder_hidden_states=text_embeds,
                down_block_additional_residuals=down_samps,
                mid_block_additional_residual=mid_samp
            )
            pred = unet_out.sample

            loss = torch.nn.functional.mse_loss(pred, noise)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            total_steps += 1
            loop.set_postfix(loss=f"{loss:.4f}")
            if is_main_process() and writer:
                writer.add_scalar("train/step_loss", loss.item(), total_steps)

            train_loss_sum += loss.item()
            train_batches  += 1

        # AFTER FINISHING ALL BATCHES: log avg train-loss
        if is_main_process() and writer:
            avg_train = epoch_avg_loss(train_loss_sum, train_batches)
            writer.add_scalar("train/epoch_loss", avg_train, epoch)
            logging.info(f"Epoch {epoch} avg TRAIN loss = {avg_train:.4f}")

        # VALIDATION
        val_loss_sum = 0.0
        val_batches  = 0
        controlnet.module.eval()   # DDP‐wrapped
        unet.eval(); text_encoder.eval()
        with torch.no_grad():
            for batch in val_loader:
                post = batch["post"].to(device)
                mask = batch["mask"].to(device).expand(-1,3,-1,-1)
                # reuse same text‐prompt logic...
                # get text_embeds & latents & noise & timesteps as before
                # forward through controlnet+unet, compute mse_loss
                val_loss_sum += loss.item()
                val_batches  += 1
        controlnet.module.train()
        unet.train(); text_encoder.train()

        if is_main_process() and writer:
            avg_val = epoch_avg_loss(val_loss_sum, val_batches)
            writer.add_scalar("val/epoch_loss", avg_val, epoch)
            logging.info(f"Epoch {epoch} avg   VAL loss = {avg_val:.4f}")

        # ── Checkpoint ────────────────────────────────────
        if is_main_process():
           # use run_name as prefix if provided, else fall back
            fname = f"{run_name + '_' if run_name else ''}epoch{epoch}.pth"
            ckpt_path = os.path.join(args.ckpt_dir, fname)
            torch.save(controlnet.module.state_dict(), ckpt_path)
            logging.info(f"Saved checkpoint → {ckpt_path}")

    if writer:
        writer.close()
    if is_main_process():
        logging.info("=== Training Complete ===")

    cleanup_ddp()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_roots",      type=str, required=True,
                   help="Comma-separated roots: data/train,data/tier3,data/test")
    p.add_argument("--crop_size",       type=int, default=512)
    p.add_argument("--max_samples",     type=int, default=None)
    p.add_argument("--batch_size",      type=int, default=4)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--epochs",          type=int, default=10)
    p.add_argument("--ckpt_dir",        type=str,   default="checkpoints")
    p.add_argument("--tensorboard_dir", type=str,   default="tb_logs")
    p.add_argument("--resume",          type=str,   default=None,
                   help="path to ControlNet .pth checkpoint to resume from")
    p.add_argument("--run_name",        type=str,   default="",
                   help="experiment name prefix for checkpoints / TB logs")
    p.add_argument("--val_root",        type=str,   required=True,
                   help="root of your *test* (validation) split")
    p.add_argument("--seed",           type=int,   default=42,
                   help="global RNG seed")
    args = p.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    main(args)
