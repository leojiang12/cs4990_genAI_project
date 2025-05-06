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

# ── 0) Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def load_pipeline(controlnet_ckpt, device="cuda"):
    logging.info(f"Loading pipeline on {device} with ckpt={controlnet_ckpt}")
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
    logging.info("Pipeline loaded.")
    return pipe


def infer_and_plot(pipe, pre_imgs, masks, metas, severities, out_path="severity_sweep.png"):
    device = pipe.device
    B      = len(pre_imgs)
    logging.info(f"Running inference on {B} samples with severities={severities}")

    # Convert to PIL & keep tensors
    pil_pre, toks_pre = [], []
    for idx, t in enumerate(pre_imgs):
        arr = ((t * 0.5 + 0.5) * 255).clamp(0,255).byte().cpu().numpy()
        arr = arr.transpose(1,2,0)
        pil = Image.fromarray(arr)
        pil_pre.append(pil)
        toks_pre.append(torch.from_numpy(arr.transpose(2,0,1)/255.0))
        logging.debug(f"Sample[{idx}] pre-image PIL size={pil.size}, tensor shape={t.shape}")

    # Build rich prompts
    prompts = []
    for idx, m in enumerate(metas):
        # location
        lnglat = m.get("features",{}).get("lng_lat", [])
        if lnglat:
            coords = lnglat[0]["properties"]
            loc = f"{coords.get('lng'):.2f}E,{coords.get('lat'):.2f}N"
        else:
            loc = "unknown location"
        # disaster
        dmeta = m.get("metadata",{})
        dtype = dmeta.get("disaster_type", dmeta.get("disaster",""))
        date  = dmeta.get("capture_date","").split("T")[0]
        p = f"{dtype} aftermath in {loc}"
        if date: p += f", on {date}"
        prompts.append(p)
        logging.debug(f"Sample[{idx}] prompt = “{p}”")

    all_rows = []
    for i in range(B):
        row = [toks_pre[i]]
        m   = masks[i].float().to(device)

        for sev in severities:
            pil_mask = Image.fromarray(
                (m * sev * 255).byte().cpu().numpy().squeeze(0)
            ).convert("L")

            if sev == 0.0:
                gen = toks_pre[i]
            else:
                # --- log right before calling the pipeline ---
                logging.debug(f"Calling pipe for sample {i}, sev={sev}")
                logging.debug(f"  prompt type={type(prompts[i])}, value=“{prompts[i]}”")
                logging.debug(f"  image type={type(pil_pre[i])}, size={pil_pre[i].size}")
                logging.debug(f"  control_image type={type(pil_mask)}, size={pil_mask.size}")
                try:
                    out = pipe(
                        prompt=prompts[i],
                        image=[pil_pre[i]],
                        controlnet_conditioning_image=[pil_mask],
                        strength=1.0 - sev,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                    ).images[0]
                except Exception as e:
                    logging.exception(f"❌ pipe(...) failed for sample {i}, sev={sev}")
                    raise

                arr  = np.array(out)
                gen  = torch.from_numpy(arr.transpose(2,0,1)/255.0)
                logging.debug(f"  → generated tensor shape={gen.shape}")

            row.append(gen)

        all_rows.extend(row)

    grid = make_grid(torch.stack(all_rows, dim=0),
                     nrow=1 + len(severities),
                     pad_value=1.0)
    plt.figure(figsize=((1+len(severities))*3, B*3))
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

    M     = min(args.max_samples, N)
    indices = (random.sample(range(N), M)
               if args.random_sample else list(range(M)))

    pre_imgs = [ ds[i]["pre"]  for i in indices ]
    masks    = [ ds[i]["mask"] for i in indices ]
    metas    = [ ds[i]["meta"] for i in indices ]
    sev_list = [float(x) for x in args.severities.split(",")]

    infer_and_plot(pipe, pre_imgs, masks, metas, sev_list, out_path=args.out)
