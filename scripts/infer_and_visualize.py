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


def infer_and_plot(pipe, pre_imgs, masks, metas, post_imgs, severities, out_path="severity_sweep.png"):
    B = len(pre_imgs)
    logging.info(f"Running inference on {B} samples with severities={severities}")

    # convert pre-disaster to PIL + tensor
    pil_pre, toks_pre = [], []
    for t in pre_imgs:
        arr = ((t * 0.5 + 0.5) * 255).clamp(0,255).byte().cpu().numpy()
        arr = arr.transpose(1,2,0)
        pil_pre.append(Image.fromarray(arr))
        toks_pre.append(torch.from_numpy(arr.transpose(2,0,1)/255.0))

    # convert post-disaster to PIL + tensor
    pil_post, toks_post = [], []
    for t in post_imgs:
        arr = ((t * 0.5 + 0.5) * 255).clamp(0,255).byte().cpu().numpy()
        arr = arr.transpose(1,2,0)
        pil_post.append(Image.fromarray(arr))
        toks_post.append(torch.from_numpy(arr.transpose(2,0,1)/255.0))

    # build simple prompts from metadata
    prompts = []
    for m in metas:
        d = m.get("metadata", {})
        dtype = d.get("disaster_type", d.get("disaster",""))
        ll = m.get("features",{}).get("lng_lat",[])
        loc = "unknown"
        if ll:
            p = ll[0].get("properties",{})
            loc = f"{p.get('lng',0):.2f}E,{p.get('lat',0):.2f}N"
        date = d.get("capture_date","").split("T")[0]
        prompt = f"{dtype} aftermath in {loc}"
        if date:
            prompt += f", on {date}"
        prompts.append(prompt)

    all_rows = []
    N = 30
    for i in range(B):
        row = [toks_pre[i]]
        for sev in severities:
            if sev == 0.0:
                gen = toks_pre[i]
            else:
                pil_mask = Image.fromarray((masks[i].float() * sev * 255)
                                           .byte().cpu().numpy().squeeze(0)).convert("L")
                out = pipe(
                    prompt=prompts[i],
                    image=[pil_pre[i]],
                    control_image=[pil_mask],
                    strength=sev,
                    num_inference_steps=N,
                    guidance_scale=10.0,
                ).images[0]
                arr = np.array(out)
                gen = torch.from_numpy(arr.transpose(2,0,1)/255.0)
            row.append(gen)

        # append the ground‑truth post‑disaster image as final column
        row.append(toks_post[i])

        all_rows.extend(row)

    # build grid: 1 pre + len(severities) + 1 post columns
    cols = 1 + len(severities) + 1
    grid = make_grid(torch.stack(all_rows, dim=0), nrow=cols, pad_value=1.0)
    plt.figure(figsize=(cols*3, B*3))
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight")
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

    pre_imgs  = [ds[i]["pre"]   for i in idxs]
    masks     = [ds[i]["mask"]  for i in idxs]
    post_imgs = [ds[i]["post"]  for i in idxs]   # ← add this line
    metas     = [ds[i]["meta"]  for i in idxs]
    sev_list  = [float(x) for x in args.severities.split(",")]

    infer_and_plot(pipe,
                    pre_imgs,
                    masks,
                    metas,
                    post_imgs,        # ← pass post_imgs here
                    sev_list,
                    out_path=args.out)
