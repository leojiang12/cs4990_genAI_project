#!/usr/bin/env python3
import os
import sys
import random
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
    StableDiffusionControlNetPipeline,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from src.datasets import XBDFullDataset  


def load_pipeline(controlnet_ckpt, device="cuda"):
    vae       = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_enc  = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    unet      = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(device)
    scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    ctrl_net  = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth").to(device)
    ctrl_net.load_state_dict(torch.load(controlnet_ckpt, map_location=device))
    ctrl_net.requires_grad_(False)

    pipe = StableDiffusionControlNetPipeline(
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
    return pipe


def infer_and_plot(pipe, pre_imgs, masks, severities, out_path="severity_sweep.png"):
    """
    pre_imgs:   list of [3×H×W] tensors in [-1,1]
    masks:      list of [1×H×W] binary tensors {0,1}
    severities: list of floats in [0,1]
    """
    device = pipe.device
    B      = len(pre_imgs)

    # Convert pre_imgs → PIL and keep the tensors for grid
    pil_pre, toks_pre = [], []
    for t in pre_imgs:
        arr = ((t * 0.5 + 0.5) * 255).clamp(0,255).byte().cpu().numpy()
        arr = arr.transpose(1,2,0)
        pil_pre.append(Image.fromarray(arr))
        toks_pre.append(torch.from_numpy(arr.transpose(2,0,1)/255.0))

    # Precompute CLIP text‑embeddings for “photo”
    prompts = ["photo"] * B
    tokens  = pipe.tokenizer(prompts, return_tensors="pt",
                              padding="max_length", truncation=True,
                              max_length=pipe.tokenizer.model_max_length).to(device)
    txt_emb = pipe.text_encoder(**tokens).last_hidden_state

    all_rows = []
    for i in range(B):
        # Start each row with the pre‑disaster image
        row = [toks_pre[i]]

        # For each severity, build a float mask and run the pipeline
        m = masks[i].float().to(device)  # [1,H,W] in {0,1}
        for sev in severities:
            # build a PIL grayscale mask whose pixel‐values=sev
            mask_float = (m * sev * 255).byte().cpu().numpy().squeeze(0)
            rgb = np.stack([mask_float]*3, axis=-1)
            pil_mask = Image.fromarray(rgb)

            # generate
            out = pipe(
                prompt_embeds=txt_emb[i:i+1],
                image=[pil_pre[i]],
                controlnet_conditioning_image=[pil_mask],
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]

            # convert generated PIL→tensor
            arr = np.array(out)
            row.append(torch.from_numpy(arr.transpose(2,0,1)/255.0))

        all_rows.extend(row)

    # make a big grid: B rows, 1 + len(severities) cols
    grid = make_grid(torch.stack(all_rows, dim=0),
                     nrow=1 + len(severities),
                     pad_value=1.0)

    plt.figure(figsize=((1+len(severities))*3, B*3))
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved severity sweep → {out_path}")


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         required=True,
                        help="path to ControlNet .pth checkpoint")
    parser.add_argument("--data_root",    required=True,
                        help="root of your dataset (labels/ & images/)")
    parser.add_argument("--max_samples",  type=int, default=4,
                        help="how many examples to visualize")
    parser.add_argument("--severities",   type=str,
                        default="0.0,0.25,0.5,0.75,1.0",
                        help="comma‑separated severity levels")
    parser.add_argument("--random_sample", action="store_true",
                        help="randomly pick samples from the dataset")
    parser.add_argument("--out",          default="results.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe   = load_pipeline(args.ckpt, device)

    # build dataset
    ds = XBDFullDataset(
        labels_root = os.path.join(args.data_root,"labels"),
        images_root = os.path.join(args.data_root,"images"),
        crop_size   = 512,
        max_samples = None,           # we’ll do our own sampling
        annotate    = False,
    )
    N = len(ds)
    if N == 0:
        raise RuntimeError("No samples found in " + args.data_root)

    # select indices
    M = min(args.max_samples, N)
    if args.random_sample:
        indices = random.sample(range(N), M)
    else:
        indices = list(range(M))

    pre_imgs = [ ds[i]["pre"]  for i in indices ]
    masks    = [ ds[i]["mask"] for i in indices ]

    # parse severities
    sev_list = [float(x) for x in args.severities.split(",")]
    infer_and_plot(pipe, pre_imgs, masks, sev_list, out_path=args.out)
