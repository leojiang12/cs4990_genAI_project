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
    StableDiffusionControlNetImg2ImgPipeline,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from src.datasets import XBDFullDataset  


def load_pipeline(controlnet_ckpt, device="cuda"):
    vae       = AutoencoderKL.from_pretrained(
                    "runwayml/stable-diffusion-v1-5", subfolder="vae"
                ).to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_enc  = CLIPTextModel.from_pretrained(
                    "openai/clip-vit-large-patch14"
                ).to(device)
    unet      = UNet2DConditionModel.from_pretrained(
                    "runwayml/stable-diffusion-v1-5", subfolder="unet"
                ).to(device)
    scheduler = DDPMScheduler.from_pretrained(
                    "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
                )
    ctrl_net  = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-depth"
                ).to(device)
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
    return pipe


def infer_and_plot(pipe, pre_imgs, masks, metas, severities, out_path="severity_sweep.png"):
    device = pipe.device
    B      = len(pre_imgs)

    # Convert to PIL & keep tensors for the final grid
    pil_pre, toks_pre = [], []
    for t in pre_imgs:
        arr = ((t * 0.5 + 0.5) * 255).clamp(0,255).byte().cpu().numpy()
        arr = arr.transpose(1,2,0)
        pil_pre.append(Image.fromarray(arr))
        toks_pre.append(torch.from_numpy(arr.transpose(2,0,1)/255.0))

    # ---- build a rich prompt from each sample's metadata ----
    prompts = []
    for m in metas:
        # location
        lnglat = m.get("features",{}).get("lng_lat", [])
        if lnglat:
            coords = lnglat[0]["properties"]
            loc = f"{coords.get('lng'):.2f}E,{coords.get('lat'):.2f}N"
        else:
            loc = "unknown location"

        # disaster fields
        dmeta = m.get("metadata",{})
        dtype = dmeta.get("disaster_type", dmeta.get("disaster",""))
        date  = dmeta.get("capture_date","").split("T")[0]
        sun   = dmeta.get("sun_elevation", None)
        off   = dmeta.get("off_nadir_angle", None)
        gsd   = dmeta.get("gsd", None)

        p = f"{dtype} aftermath in {loc}"
        if date: p += f", on {date}"
        if sun:  p += f", sunny {sun:.1f}°"
        if off:  p += f", off‑nadir {off:.1f}°"
        if gsd:  p += f", {gsd:.2f} m/px"
        prompts.append(p)
    # -----------------------------------------------------------

    all_rows = []
    for i in range(B):
        row = [toks_pre[i]]
        m   = masks[i].float().to(device)

        for sev in severities:
            mask_arr = (m * sev * 255).byte().cpu().numpy().squeeze(0)
            pil_mask = Image.fromarray(mask_arr).convert("L")

            if sev == 0.0:
                gen = toks_pre[i]
            else:
                out = pipe(
                    prompt=prompts[i],
                    init_image=[pil_pre[i]],
                    controlnet_conditioning_image=[pil_mask],
                    strength=1.0 - sev,           # ← sev here
                    num_inference_steps=30,
                    guidance_scale=7.5,
                ).images[0]
                arr = np.array(out)
                gen = torch.from_numpy(arr.transpose(2,0,1)/255.0)

            row.append(gen)

        all_rows.extend(row)

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
    pipe   = load_pipeline(args.ckpt, device)

    ds = XBDFullDataset(
        labels_root = os.path.join(args.data_root,"labels"),
        images_root = os.path.join(args.data_root,"images"),
        crop_size   = 512,
        max_samples = None,
        annotate    = False,
    )
    N = len(ds)
    if N == 0:
        raise RuntimeError("No samples found in " + args.data_root)

    M = min(args.max_samples, N)
    indices = (random.sample(range(N), M)
               if args.random_sample else list(range(M)))

    pre_imgs = [ ds[i]["pre"]  for i in indices ]
    masks    = [ ds[i]["mask"] for i in indices ]
    metas    = [ ds[i]["meta"] for i in indices ]
    sev_list = [float(x) for x in args.severities.split(",")]

    infer_and_plot(pipe, pre_imgs, masks, metas, sev_list, out_path=args.out)
