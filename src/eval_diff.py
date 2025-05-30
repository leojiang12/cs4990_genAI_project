#!/usr/bin/env python3
import os
import argparse

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

from src.datasets import XBDPairDataset


def load_models(device, ckpt_path):
    # 1) Load pretrained Stable‐Diffusion pieces (frozen)
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae"
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet"
    ).to(device)
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14"
    ).to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    scheduler = DDPMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
    )

    # 2) Load your fine-tuned ControlNet
    if os.path.isdir(ckpt_path):
        # you saved via controlnet.save_pretrained(...)
        controlnet = ControlNetModel.from_pretrained(ckpt_path).to(device)
    else:
        # assume ckpt_path is a .pth state-dict
        # pick a matching base ControlNet (e.g. the one you originally fine-tuned)
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny"
        )
        sd = torch.load(ckpt_path, map_location="cpu")
        # if you wrapped it in {"state_dict": …}
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        controlnet.load_state_dict(sd)
        controlnet = controlnet.to(device)

    # 3) freeze everything but ControlNet
    for p in vae.parameters():
        p.requires_grad = False
    for p in unet.parameters():
        p.requires_grad = False
    for p in text_encoder.parameters():
        p.requires_grad = False

    return vae, tokenizer, text_encoder, unet, controlnet, scheduler


@torch.no_grad()
def generate(
    controlnet, unet, scheduler, vae, tokenizer, text_encoder, pre, severity, device
):
    B, _, H, W = pre.shape
    # 1) encode “pre” into latents
    latents = vae.encode(pre).latent_dist.sample() * vae.config.scaling_factor
    # 2) add noise
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (B,), device=device
    )
    noise = torch.randn_like(latents)
    noisy = scheduler.add_noise(latents, noise, timesteps)
    # 3) prepare text embedding (“photo” prompt)
    tokens = tokenizer(
        ["photo"] * B,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(device)
    txt_emb = text_encoder(**tokens).last_hidden_state
    # 4) control map from “severity”
    control_map = severity.view(B, 1, 1, 1).expand(B, 3, H, W)
    # 5) run ControlNet & Unet
    ctrl = controlnet(
        noisy,
        timestep=timesteps,
        encoder_hidden_states=txt_emb,
        controlnet_cond=control_map,
    )
    unet_out = unet(
        noisy,
        timestep=timesteps,
        encoder_hidden_states=txt_emb,
        down_block_additional_residuals=ctrl.down_block_res_samples,
        mid_block_additional_residual=ctrl.mid_block_res_sample,
    )
    # 6) predict noise & denoise one step
    pred_noise = unet_out.sample
    latents_denoised = scheduler.step(pred_noise, timesteps, noisy).prev_sample
    # 7) decode back to RGB
    gen = vae.decode(latents_denoised / vae.config.scaling_factor).sample
    return gen.clamp(0, 1)


def plot_and_save(pre, post, gen, loss, out_path):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Pre", f"Post\nLoss={loss:.4f}", "Generated"]
    for ax, img, title in zip(axs, [pre, post, gen], titles):
        #ax.imshow(img.permute(1, 2, 0).cpu().numpy())
        # unnormalize from [-1,1] → [0,1]
        disp = (img * 0.5) + 0.5
        ax.imshow(disp.permute(1, 2, 0).cpu().numpy())
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="path to your ControlNet checkpoint (folder or .pth)",
    )
    p.add_argument("--labels_dir", type=str, required=True)
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="how many val samples to visualize",
    )
    p.add_argument("--out_dir", type=str, default="eval_outputs")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # load all models
    vae, tokenizer, txt_enc, unet, ctrl, sched = load_models(device, args.ckpt)

    # prepare your val dataset
    ds = XBDPairDataset(
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        crop_size=512,
        max_samples=None,
        annotate=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    for idx, batch in enumerate(loader):
        if idx >= args.num_samples:
            break

                # — save the full-color pre & post disaster images —
        post_json = ds.post_jsons[idx]
        # compute the relative .png path from the JSON path
        rel     = os.path.relpath(post_json, start=os.path.dirname(ds.post_jsons[0]))
        png_rel = rel.replace("_post_disaster.json", "_post_disaster.png")
        # load & save full-res post
        full_post = Image.open(os.path.join(args.images_dir, png_rel)).convert("RGB")
        full_post.save(os.path.join(args.out_dir, f"full_post_{idx:03d}.png"))
        # load & save full-res pre
        full_pre = Image.open(
            os.path.join(args.images_dir, png_rel.replace("_post_disaster.png", "_pre_disaster.png"))
        ).convert("RGB")
        full_pre.save(os.path.join(args.out_dir, f"full_pre_{idx:03d}.png"))


        pre = batch["pre"].to(device)  # [B,3,H,W]
        post = batch["post"].to(device)  # [B,3,H,W]
        severity = batch["severity"].to(device).unsqueeze(-1)  # [B,1]

        gen = generate(ctrl, unet, sched, vae, tokenizer, txt_enc, pre, severity, device)
        loss = F.mse_loss(gen, post).item()

        out_path = os.path.join(args.out_dir, f"sample_{idx:03d}.png")
        plot_and_save(pre[0], post[0], gen[0], loss, out_path)
        print(f"Saved ▶ {out_path}")


if __name__ == "__main__":
    main()
