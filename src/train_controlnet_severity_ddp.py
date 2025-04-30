# src/train_controlnet_severity_ddp.py

import os
import argparse
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

from src.datasets import XBDPairDataset

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
    # ── DDP setup ──────────────────────────────────────
    setup_ddp(args)
    device = torch.device(f"cuda:{args.local_rank}")

    # ── 1) Load & freeze models ────────────────────────
    vae     = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    unet    = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(device)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth").to(device)
    scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    for p in vae.parameters():         p.requires_grad = False
    for p in unet.parameters():        p.requires_grad = False
    for p in text_encoder.parameters():p.requires_grad = False
    for p in controlnet.parameters():  p.requires_grad = True

    # wrap ControlNet for DDP
    controlnet = DDP(controlnet, device_ids=[args.local_rank], output_device=args.local_rank)

    # ── 2) Dataset & Dataloader ────────────────────────
    ds = XBDPairDataset(
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        crop_size=args.crop_size,
        max_samples=args.max_samples,
        annotate=False
    )
    if is_main_process():
        print(f"Dataset samples found: {len(ds)}")

    sampler = DistributedSampler(ds, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    loader  = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    if is_main_process():
        print(f"DDP Training on {args.world_size} GPUs, world_size={args.world_size}")

    # ── 3) Optimizer ────────────────────────────────────
    optimizer = AdamW(controlnet.parameters(), lr=args.lr)

    # ── 4) Training Loop ────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        loop = tqdm(loader, desc=f"Epoch [{epoch}/{args.epochs}]", disable=not is_main_process())

        for batch in loop:
            pre = batch["pre"].to(device)                      # [B,3,H,W]
            post = batch["post"].to(device)                    # [B,3,H,W]
            severity = batch["severity"].to(device).unsqueeze(-1)  # [B,1]

            # a) Build severity map and tile to 3 channels for ControlNet
            B,_,H,W = pre.shape
            # severity is [B,1]; reshape to [B,1,1,1] then expand to [B,3,H,W]
            control_map = severity.view(B,1,1,1).expand(B,3,H,W)

            # b) Dummy text tokens (“photo”) 
            prompts = ["photo"] * B
            tokens  = tokenizer(
                prompts, return_tensors="pt",
                padding="max_length", truncation=True,
                max_length=tokenizer.model_max_length
            ).to(device)
            text_embeds = text_encoder(**tokens).last_hidden_state

            # c) Encode post → latents
            with torch.no_grad():
                latents = vae.encode(post).latent_dist.sample() * vae.config.scaling_factor

            # d) Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps,
                (B,), device=device
            )
            noisy = scheduler.add_noise(latents, noise, timesteps)

            # e) ControlNet forward (same as before)
            ctrl_outputs = controlnet(
                noisy,
                timestep=timesteps,
                encoder_hidden_states=text_embeds,
                controlnet_cond=control_map,
            )
            down_samples  = ctrl_outputs.down_block_res_samples       # list of [B,C_i,H,W]
            mid_sample    = ctrl_outputs.mid_block_res_sample         # one [B,C_mid,H,W]

            # f) UNet forward (hook in the ControlNet residuals)
            model_out = unet(
                noisy,
                timestep=timesteps,
                encoder_hidden_states=text_embeds,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample,
            )
            # in diffusers v2.x this returns a UNetOutput with `.sample`
            model_pred = model_out.sample


            # g) Loss & backward
            loss = torch.nn.functional.mse_loss(model_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        # ── Checkpointing ───────────────────────────────
        if is_main_process():
            ckpt = os.path.join(args.ckpt_dir, f"controlnet_epoch{epoch}.pth")
            torch.save(controlnet.module.state_dict(), ckpt)
            print(f"Saved ControlNet checkpoint: {ckpt}")

    if is_main_process():
        print("=== Training Complete ===")

    from torchinfo import summary

    if is_main_process():
        # Summarize the *combined* pipeline:
        summary(
            {
                "latents":    torch.randn(1, unet.in_channels, H, W).to(device),
                "timesteps":  torch.randint(0, scheduler.num_train_timesteps, (1,), device=device),
                "text_embeds":torch.randn(1, tokenizer.model_max_length, text_encoder.config.hidden_size).to(device),
                "ctrl_map":   torch.rand(1, 1, H, W).to(device),
            },
            model={
                "ControlNet": controlnet.module, 
                "UNet":       unet
            },
            col_names=["input_size", "output_size", "num_params"],
            depth=5
        )

    cleanup_ddp()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels_dir",    type=str, required=True)
    p.add_argument("--images_dir",    type=str, required=True)
    p.add_argument("--crop_size",     type=int, default=512)
    p.add_argument("--max_samples",   type=int, default=None)
    p.add_argument("--batch_size",    type=int, default=4)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--epochs",        type=int, default=10)
    p.add_argument("--ckpt_dir",      type=str, default="checkpoints")
    args = p.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    main(args)
