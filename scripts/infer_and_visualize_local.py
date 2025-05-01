#!/usr/bin/env python3
# scripts/infer_and_visualize_local.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import CLIPTokenizer

# ← use the recursive dataset so you actually see all ~20 k samples
from src.datasets import XBDFullDataset  


def load_pipeline(controlnet_ckpt: str, device="cuda"):
    """
    Load vanilla SD ControlNet pipeline, swap in your fine-tuned weights,
    and enable attention slicing.
    """
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth"),
        safety_checker=None,               # ← disable safety
        feature_extractor=None,            # ← disable related extractor
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)

    # overwrite with your checkpoint
    state = torch.load(controlnet_ckpt, map_location=device)
    pipe.controlnet.load_state_dict(state)

    # reduce VRAM
    pipe.enable_attention_slicing()
    return pipe


def make_severity_map(severity: float, width: int, height: int) -> Image.Image:
    """
    Return a constant-gray RGB PIL image of size (width×height)
    where each channel = int(severity * 255).
    """
    val = int(severity * 255)
    arr = np.full((height, width, 3), val, dtype=np.uint8)
    return Image.fromarray(arr)


def infer_and_plot(pipe, pre_imgs, severities, out_path="severity_sweep.png"):
    device = pipe.device
    batch  = len(pre_imgs)

    # convert your normalized [-1,1] tensors back to PIL
    pil_pre = []
    for t in pre_imgs:
        # t ∈ [–1,1], shape [3,H,W]
        img = ((t * 0.5 + 0.5) * 255).clamp(0,255).cpu().numpy().astype(np.uint8)
        img = img.transpose(1,2,0)
        pil_pre.append(Image.fromarray(img))

    # use the same prompt for all
    prompts = ["photo"] * batch

    # figure out spatial dims for control‐map
    cf_size = pipe.vae.config.sample_size  # usually 64 or 128

    rows = []
    for sev in severities:
        # build one constant PIL per sample
        ctrl_images = [make_severity_map(sev, cf_size, cf_size) for _ in range(batch)]

        # run the pipeline
        out = pipe(
            prompt=prompts,
            image=pil_pre,
            controlnet_conditioning_image=ctrl_images,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images  # List[PIL]

        # convert back into tensors for make_grid
        toks = []
        for im in out:
            arr = np.array(im).astype(np.float32) / 255.0  # H×W×3
            toks.append(torch.from_numpy(arr).permute(2,0,1))  # 3×H×W
        rows.append(torch.stack(toks, dim=0))

    # grid: rows × batch
    grid = make_grid(torch.cat(rows, dim=0), nrow=batch, pad_value=1.0)
    plt.figure(figsize=(batch*3, len(severities)*3))
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.axis("off")
    plt.title("Rows = severities, Cols = samples")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved visualization to {out_path}")


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       required=True,
                        help="your ControlNet .pth checkpoint")
    parser.add_argument("--data_root",  required=True,
                        help="root dir with `labels/` & `images/` subfolders")
    parser.add_argument("--max_samples",type=int, default=4,
                        help="how many examples to pull (default=4)")
    parser.add_argument("--out",        default="severity_sweep.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe   = load_pipeline(args.ckpt, device)

    # recursive loader now
    ds = XBDFullDataset(
        labels_root = os.path.join(args.data_root, "labels"),
        images_root = os.path.join(args.data_root, "images"),
        crop_size   = 512,
        max_samples = args.max_samples,
        annotate    = False,
    )
    if len(ds)==0:
        raise RuntimeError(f"No samples found under {args.data_root}")

    pre_imgs   = [ ds[i]["pre"] for i in range(len(ds)) ]
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]

    infer_and_plot(pipe, pre_imgs, severities, out_path=args.out)
    print("Done!")
