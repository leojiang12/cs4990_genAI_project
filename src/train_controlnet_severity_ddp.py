#!/usr/bin/env python3
import os, re, random, argparse, logging
from datetime import datetime
from tqdm import tqdm

import torch, torch.distributed as dist, numpy as np
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
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from src.datasets import XBDFullDataset

def setup_ddp():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()

def cleanup_ddp():
    dist.destroy_process_group()

def is_main():
    return dist.get_rank() == 0

def tensor_to_np(x: torch.Tensor):
    # x: [B, C, H, W]
    arr = x.cpu().numpy().transpose(0,2,3,1)  # → [B, H, W, C]
    arr = (arr * 255).astype(np.uint8)
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_roots",      required=True,
                   help="Comma‑sep train roots")
    p.add_argument("--val_root",        required=True,
                   help="Validation root")
    p.add_argument("--crop_size",       type=int,   default=512)
    p.add_argument("--max_samples",     type=int,   default=None)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--epochs",          type=int,   default=10)
    p.add_argument("--ckpt_dir",        type=str,   default="checkpoints")
    p.add_argument("--tensorboard_dir", type=str,   default="tb_logs")
    p.add_argument("--resume",          type=str,   default=None)
    p.add_argument("--run_name",        type=str,   default="")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--no_controlnet", action="store_true",
                   help="Skip ControlNet (baseline)")
    args = p.parse_args()

    # ── reproducibility ────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    local_rank, rank, world = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # ── models ─────────────────────────────────────────
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5",
                                        subfolder="vae").to(device)
    clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_txt = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                subfolder="unet").to(device)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth").to(device)
    scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5",
                                              subfolder="scheduler")

    # freeze everything except ControlNet
    for m in (vae, clip_txt, unet):
        for p in m.parameters(): p.requires_grad = False
    for p in controlnet.parameters(): p.requires_grad = True

    controlnet = DDP(controlnet, device_ids=[local_rank])

    # ── data loaders ────────────────────────────────────
    # validation
    val_ds = XBDFullDataset(os.path.join(args.val_root,"labels"),
                            os.path.join(args.val_root,"images"),
                            crop_size=args.crop_size, annotate=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            sampler=DistributedSampler(val_ds, world, rank, shuffle=False),
                            num_workers=4, pin_memory=True)

    # training
    roots = [r.strip() for r in args.data_roots.split(",")]
    parts = []
    for rt in roots:
        parts.append(XBDFullDataset(os.path.join(rt,"labels"),
                                    os.path.join(rt,"images"),
                                    crop_size=args.crop_size,
                                    max_samples=args.max_samples,
                                    annotate=False))
    train_ds = ConcatDataset(parts)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=DistributedSampler(train_ds, world, rank, shuffle=True),
                              num_workers=4, pin_memory=True)

    # ── TensorBoard ─────────────────────────────────────
    writer = None
    if is_main():
        run_id = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_sub = os.path.join(args.tensorboard_dir, run_id)
        os.makedirs(tb_sub, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_sub)
        # log hparams
        writer.add_hparams({
            "lr": args.lr,
            "batch_size": args.batch_size,
            "run_name": run_id,
            "seed": args.seed,
            "no_controlnet": args.no_controlnet,
        }, {})

    os.makedirs(args.ckpt_dir, exist_ok=True)
    optimizer = AdamW(controlnet.parameters(), lr=args.lr)

    # resume
    start_epoch = 1
    if args.resume:
        sd = torch.load(args.resume, map_location="cpu")
        controlnet.module.load_state_dict(sd)
        m = re.search(r"epoch(\d+)\.pth", args.resume)
        if m: start_epoch = int(m.group(1))+1

    # ── training loop ───────────────────────────────────
    for ep in range(start_epoch, args.epochs+1):
        controlnet.train(); unet.train(); clip_txt.train()
        train_loss = 0.0; train_batches = 0
        loop = tqdm(train_loader, disable=not is_main(), desc=f"Epoch {ep}/{args.epochs}")
        for batch in loop:
            pre = batch["pre"].to(device)
            post = batch["post"].to(device)
            mask = batch["mask"].to(device)  # [B,1,H,W]

            # prompt tokens
            # … you can inline your metadata→prompt code …
            prompts = ["disaster aftermath"]*pre.shape[0]
            toks = clip_tok(prompts, padding="max_length", truncation=True,
                            return_tensors="pt").to(device)
            txt_emb = clip_txt(**toks).last_hidden_state

            with torch.no_grad():
                lat = vae.encode(post).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(lat)
            ts = torch.randint(0, scheduler.config.num_train_timesteps,
                               (lat.size(0),), device=device)
            noisy = scheduler.add_noise(lat, noise, ts)

            if args.no_controlnet:
                unet_out = unet(noisy, timestep=ts,
                                encoder_hidden_states=txt_emb)
            else:
                ctrl_out = controlnet(noisy, timestep=ts,
                                      encoder_hidden_states=txt_emb,
                                      controlnet_cond=mask.expand(-1,3,-1,-1))
                unet_out = unet(noisy, timestep=ts,
                                encoder_hidden_states=txt_emb,
                                down_block_additional_residuals=ctrl_out.down_block_res_samples,
                                mid_block_additional_residual=ctrl_out.mid_block_res_sample)

            pred = unet_out.sample
            loss = torch.nn.functional.mse_loss(pred, noise)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            train_loss += loss.item(); train_batches += 1
            loop.set_postfix(loss=f"{loss:.4f}")

        avg_train = train_loss/train_batches
        if is_main() and writer:
            writer.add_scalar("train/epoch_mse", avg_train, ep)

        # ── validation ─────────────────────────────────────
        controlnet.eval(); unet.eval(); clip_txt.eval()
        val_mse = val_psnr = val_ssim = 0.0
        vb = 0
        with torch.no_grad():
            for batch in val_loader:
                post = batch["post"].to(device)
                mask = batch["mask"].expand(-1,3,-1,-1).to(device)
                toks = clip_tok([""]*post.shape[0], return_tensors="pt",
                                padding="max_length", truncation=True).to(device)
                txt_emb = clip_txt(**toks).last_hidden_state

                lat = vae.encode(post).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(lat)
                ts = torch.randint(0, scheduler.config.num_train_timesteps,
                                   (lat.size(0),), device=device)
                noisy = scheduler.add_noise(lat, noise, ts)

                if args.no_controlnet:
                    out = unet(noisy, timestep=ts,
                               encoder_hidden_states=txt_emb).sample
                else:
                    c = controlnet(noisy, timestep=ts,
                                   encoder_hidden_states=txt_emb,
                                   controlnet_cond=mask)
                    out = unet(noisy, timestep=ts,
                               encoder_hidden_states=txt_emb,
                               down_block_additional_residuals=c.down_block_res_samples,
                               mid_block_additional_residual=c.mid_block_res_sample).sample

                mse = torch.nn.functional.mse_loss(out, noise).item()
                # decode to images
                # FIX: post is [B,3,H,W], so just pass (post+1)/2 directly
                gt_img = tensor_to_np((post+1)/2)[0]
                pred_img = tensor_to_np(((vae.decode(out/vae.config.scaling_factor)+1)/2))[0]
                psnr = compute_psnr(gt_img, pred_img, data_range=255)
                ssim = compute_ssim(gt_img, pred_img, multichannel=True, data_range=255)

                val_mse  += mse
                val_psnr += psnr
                val_ssim += ssim
                vb += 1

        avg_val_mse  = val_mse/vb
        avg_val_psnr = val_psnr/vb
        avg_val_ssim = val_ssim/vb
        if is_main() and writer:
            writer.add_scalar("val/epoch_mse",  avg_val_mse, ep)
            writer.add_scalar("val/epoch_psnr", avg_val_psnr, ep)
            writer.add_scalar("val/epoch_ssim", avg_val_ssim, ep)

        # ── checkpoint ────────────────────────────────────
        if is_main():
            run = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
            fname = f"{run}_epoch{ep}.pth"
            torch.save(controlnet.module.state_dict(),
                       os.path.join(args.ckpt_dir, fname))
            logging.info(f"Saved checkpoint → {fname}")

    if is_main(): writer.close()
    cleanup_ddp()

if __name__=="__main__":
    main()
