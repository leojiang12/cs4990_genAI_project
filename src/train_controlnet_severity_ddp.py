#!/usr/bin/env python3
import os
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
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.tensorboard_dir)
        logging.info(f"TensorBoard logs → {args.tensorboard_dir}")

    # ── 4) Optimizer & optional resume ──────────────────
    optimizer   = AdamW(controlnet.parameters(), lr=args.lr)
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

    # ── 5) Training Loop ────────────────────────────────
    total_steps = 0
    for epoch in range(start_epoch, args.epochs + 1):
        sampler.set_epoch(epoch)
        loop = tqdm(loader, desc=f"Epoch [{epoch}/{args.epochs}]", disable=not is_main_process())
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
                writer.add_scalar("train/loss", loss.item(), total_steps)

        # ── Checkpoint ────────────────────────────────────
        if is_main_process():
            ckpt_path = os.path.join(args.ckpt_dir, f"controlnet_epoch{epoch}.pth")
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
    args = p.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    main(args)
