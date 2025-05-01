#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import CLIPTokenizer

# ← use the recursive dataset
from src.datasets import XBDFullDataset  


def load_pipeline(controlnet_ckpt: str, device="cuda"):
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth"),
        safety_checker=None,
        feature_extractor=None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)

    state = torch.load(controlnet_ckpt, map_location=device)
    pipe.controlnet.load_state_dict(state)
    pipe.enable_attention_slicing()
    return pipe


def infer_and_plot(pipe, pre_imgs, masks, out_path="severity_sweep.png"):
    device = pipe.device
    batch  = len(pre_imgs)

    # pre_imgs → PIL
    pil_pre = []
    for t in pre_imgs:
        img = ((t * 0.5 + 0.5) * 255).clamp(0,255).byte().cpu().numpy()
        img = img.transpose(1,2,0)
        pil_pre.append(Image.fromarray(img))

    # masks → 3‐channel PIL
    pil_mask = []
    for m in masks:
        arr = (m.squeeze(0).mul(255).byte().cpu().numpy())
        rgb = np.stack([arr]*3, axis=-1)
        pil_mask.append(Image.fromarray(rgb))

    prompts = ["photo"] * batch
    cf_size = pipe.vae.config.sample_size

    # run once
    outs = pipe(
        prompt=prompts,
        image=pil_pre,
        controlnet_conditioning_image=pil_mask,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images

    # assemble grid
    toks = [torch.from_numpy(np.array(im).transpose(2,0,1)/255.0) for im in outs]
    grid = make_grid(torch.stack(toks, dim=0), nrow=batch, pad_value=1.0)
    plt.figure(figsize=(batch*3,3))
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved visualization to {out_path}")


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       required=True,
                        help="your ControlNet .pth checkpoint")
    parser.add_argument("--data_root",  required=True,
                        help="root dir with `labels/` & `images/`")
    parser.add_argument("--max_samples",type=int, default=4,
                        help="how many examples to pull (default=4)")
    parser.add_argument("--out",        default="severity_sweep.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe   = load_pipeline(args.ckpt, device)

    ds = XBDFullDataset(
        labels_root=os.path.join(args.data_root,"labels"),
        images_root=os.path.join(args.data_root,"images"),
        crop_size=512,
        max_samples=args.max_samples,
        annotate=False,
    )
    if len(ds)==0:
        raise RuntimeError("No samples found under data_root")

    pre_imgs = [ ds[i]["pre"]  for i in range(len(ds)) ]
    masks    = [ ds[i]["mask"] for i in range(len(ds)) ]
    infer_and_plot(pipe, pre_imgs, masks, out_path=args.out)
    print("Done!")
