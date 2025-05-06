#!/usr/bin/env python3
import os
import sys
import random
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid

# ensure src/ is on your PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from src.datasets import XBDFullDataset  

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_pipeline(controlnet_ckpt, device="cuda"):
    logging.info(f"Loading Img2Img+ControlNet pipeline on {device} with ckpt={controlnet_ckpt}")
    vae       = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_enc  = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    unet      = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(device)
    scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    ctrl_net  = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth").to(device)
    ctrl_net.load_state_dict(torch.load(controlnet_ckpt, map_location=device))
    ctrl_net.requires_grad_(False)

    pipe = StableDiffusionControlNetImg2ImgPipeline(
        vae=vae,
        text_encoder=text_enc,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=ctrl_net,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    ).to(device)
    pipe.enable_model_cpu_offload()

    logging.info("Pipeline loaded.")
    return pipe


def infer_and_plot(pipe, pre_imgs, masks, metas, severities, out_path="severity_sweep.png"):
    """
    pipe:           StableDiffusionControlNetImg2ImgPipeline
    pre_imgs:       list of [3,H,W] tensors in [-1..1]
    masks:          list of [1,H,W] float⊂[0,1]
    metas:          list of metadata dicts
    severities:     list of floats e.g. [0.0, 0.25, …, 1.0]
    """
    B = len(pre_imgs)
    K = len(severities)

    # — Convert pre_imgs once to PIL
    pil_pres = []
    for t in pre_imgs:
        arr = ((t * 0.5 + 0.5) * 255).clamp(0,255).byte().cpu().numpy()
        pil_pres.append(Image.fromarray(arr.transpose(1,2,0)))

    # — Build textual labels
    def make_label(meta):
        dt = meta.get("metadata",{}).get("disaster_type","")
        ll = meta.get("features",{}).get("lng_lat",[])
        if ll:
            p = ll[0]["properties"]
            loc = f"{p.get('lng',0):.2f}E,{p.get('lat',0):.2f}N"
        else:
            loc = "unknown"
        return f"{dt}\n{loc}"

    labels = [make_label(m) for m in metas]

    # — Quick helper: given idx i, severity s → run the network
    def generate(i, s):
        mask_arr = (masks[i] * s * 255).byte().cpu().numpy().squeeze()
        control = Image.fromarray(mask_arr).convert("L")
        out = pipe(
            prompt   = f"{labels[i].splitlines()[0]} aftermath",  # or reuse your real prompt list
            image    = [pil_pres[i]],
            control_image = [control],
            strength = s,
            num_inference_steps = 30,
            guidance_scale = 7.5,
        ).images[0]
        return out

    # — Layout the grid
    fig, axes = plt.subplots(
        nrows = B+1, ncols = K+1,
        figsize = ((K+1)*2.5, (B+1)*2.5),
        gridspec_kw = {"wspace":0.1, "hspace":0.1},
    )

    # Header row
    axes[0,0].axis("off")
    for j, s in enumerate(severities, start=1):
        ax = axes[0,j]
        ax.axis("off")
        ax.set_title(f"{s:.2f}", pad=4)

    # Fill in each sample row
    for i in range(B):
        # metadata label in col 0
        ax = axes[i+1,0]
        ax.axis("off")
        ax.text(0.5, 0.5, labels[i],
                ha="center", va="center", fontsize=9)

        # col 1: severity = 0 “identity”
        ax = axes[i+1,1]
        ax.imshow(pil_pres[i])
        ax.axis("off")

        # cols 2…K+1: generated
        for j, s in enumerate(severities[1:], start=2):
            img = generate(i, s)
            ax = axes[i+1,j]
            ax.imshow(img)
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    logging.info(f"Saved severity sweep → {out_path}")


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        required=True,
                        help="path to ControlNet .pth checkpoint")
    parser.add_argument("--data_root",   required=True,
                        help="root of your dataset (labels/ & images/)")
    parser.add_argument("--max_samples", type=int, default=4,
                        help="how many examples to visualize")
    parser.add_argument("--severities",  type=str,
                        default="0.0,0.25,0.5,0.75,1.0",
                        help="comma‑separated severity levels")
    parser.add_argument("--random_sample", action="store_true",
                        help="randomly pick samples")
    parser.add_argument("--out",         default="results.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipeline(args.ckpt, device)

    ds = XBDFullDataset(
        labels_root = os.path.join(args.data_root, "labels"),
        images_root = os.path.join(args.data_root, "images"),
        crop_size   = 512,
        max_samples = None,
        annotate    = False,
    )
    N = len(ds)
    if N == 0:
        raise RuntimeError("No samples found in " + args.data_root)

    M = min(args.max_samples, N)
    idxs = (random.sample(range(N), M)
            if args.random_sample else list(range(M)))

    pre_imgs = [ds[i]["pre"]  for i in idxs]
    masks    = [ds[i]["mask"] for i in idxs]
    metas    = [ds[i]["meta"] for i in idxs]
    sev_list = [float(x) for x in args.severities.split(",")]

    infer_and_plot(pipe, pre_imgs, masks, metas, sev_list, out_path=args.out)
