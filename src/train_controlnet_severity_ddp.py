import os
import argparse
import logging
from datetime import datetime
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

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

    # ── Logging setup ──────────────────────────────────
    if is_main_process():
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        log_filename = os.path.join(args.tensorboard_dir,
            f"train_{datetime.now():%Y%m%d_%H%M%S}.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Starting DDP training on {args.world_size} GPUs")

        # TensorBoard
        tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)
    else:
        tb_writer = None

    # ── 1) Load & freeze models ────────────────────────
    vae          = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
    tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    unet         = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(device)
    controlnet   = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth").to(device)
    scheduler    = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    for p in vae.parameters():         p.requires_grad = False
    for p in unet.parameters():        p.requires_grad = False
    for p in text_encoder.parameters():p.requires_grad = False
    for p in controlnet.parameters():  p.requires_grad = True

    controlnet = DDP(controlnet, device_ids=[args.local_rank], output_device=args.local_rank)

    # ── 2) Dataset & Dataloader ────────────────────────
    ds = XBDPairDataset(
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        crop_size=args.crop_size,
        max_samples=args.max_samples,
        annotate=False
    )
    sampler = DistributedSampler(ds, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    loader  = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                         num_workers=4, pin_memory=True)

    if is_main_process():
        logging.info(f"Dataset samples found: {len(ds)}")

    # ── 3) Optimizer ────────────────────────────────────
    optimizer = AdamW(controlnet.parameters(), lr=args.lr)

    # ── 4) Training Loop ────────────────────────────────
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        if is_main_process():
            logging.info(f"Epoch {epoch}/{args.epochs} start")

        loop = tqdm(loader, desc=f"Epoch [{epoch}/{args.epochs}]", disable=not is_main_process())
        for batch in loop:
            pre      = batch["pre"].to(device)
            post     = batch["post"].to(device)
            severity = batch["severity"].to(device).unsqueeze(-1)

            # a) severity‐map → 3 ch
            B,_,H,W = pre.shape
            control_map = severity.view(B,1,1,1).expand(B,3,H,W)

            # b) text embeddings
            prompts = ["photo"] * B
            tokens  = tokenizer(prompts, return_tensors="pt",
                                padding="max_length", truncation=True,
                                max_length=tokenizer.model_max_length).to(device)
            text_embeds = text_encoder(**tokens).last_hidden_state

            # c) encode post → latents
            with torch.no_grad():
                latents = vae.encode(post).latent_dist.sample() * vae.config.scaling_factor

            # d) noise + timesteps
            noise     = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device)
            noisy     = scheduler.add_noise(latents, noise, timesteps)

            # e) ControlNet
            ctrl_outs     = controlnet(noisy, timestep=timesteps,
                                       encoder_hidden_states=text_embeds,
                                       controlnet_cond=control_map)
            down_res      = ctrl_outs.down_block_res_samples
            mid_res       = ctrl_outs.mid_block_res_sample

            # f) UNet
            unet_out = unet(noisy, timestep=timesteps,
                            encoder_hidden_states=text_embeds,
                            down_block_additional_residuals=down_res,
                            mid_block_additional_residual=mid_res)
            model_pred = unet_out.sample

            # g) loss & backward
            loss = torch.nn.functional.mse_loss(model_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            if is_main_process():
                tb_writer.add_scalar("train/mse_loss", loss.item(), global_step)
            loop.set_postfix(loss=loss.item())

            global_step += 1

        if is_main_process():
            ckpt = os.path.join(args.ckpt_dir, f"controlnet_epoch{epoch}.pth")
            torch.save(controlnet.module.state_dict(), ckpt)
            logging.info(f"Saved checkpoint {ckpt}")

    if is_main_process():
        logging.info("=== TRAINING COMPLETE ===")
        tb_writer.close()

    cleanup_ddp()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels_dir",       type=str, required=True)
    p.add_argument("--images_dir",       type=str, required=True)
    p.add_argument("--crop_size",        type=int, default=512)
    p.add_argument("--max_samples",      type=int, default=None)
    p.add_argument("--batch_size",       type=int, default=4)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--epochs",           type=int, default=10)
    p.add_argument("--ckpt_dir",         type=str,   default="checkpoints")
    p.add_argument("--tensorboard_dir",  type=str,   default="tensorboard")
    args = p.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    main(args)
