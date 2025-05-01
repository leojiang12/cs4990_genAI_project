#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# so that `src/` is on your PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

# ← use the recursive dataset
from src.datasets import XBDFullDataset  


def load_pipeline(controlnet_ckpt, device="cuda"):
    # 1) Base SD components
    vae       = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_enc  = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    unet      = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(device)

    # 2) Scheduler
    scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    # 3) ControlNet + your fine-tuned weights
    ctrl_net = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth").to(device)
    ctrl_net.load_state_dict(torch.load(controlnet_ckpt, map_location=device))
    ctrl_net.requires_grad_(False)

    # 4) Build the pipeline
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

    pipe.enable_model_cpu_offload()  # optional
    return pipe


def make_severity_map(severity, H, W, device):
    """
    Build a [1×1×H×W] tensor full of `severity` in [0,1].
    """
    return torch.full((1, 1, H, W), float(severity), device=device)


def infer_and_plot(pipe, pre_imgs, severities, out_path="results.png"):
    device = pipe.device
    batch  = len(pre_imgs)

    # 1) tokenize & text-encode
    prompts     = ["photo"] * batch
    text_inputs = pipe.tokenizer(
        prompts, return_tensors="pt",
        padding="max_length", truncation=True,
        max_length=pipe.tokenizer.model_max_length,
    ).to(device)
    txt_embeds = pipe.text_encoder(**text_inputs).last_hidden_state

    # 2) figure out latent-shape by encoding a dummy batch
    with torch.no_grad():
        dummy_latents = pipe.vae.encode(torch.stack(pre_imgs).to(device)) \
                            .latent_dist.sample() * pipe.vae.config.scaling_factor
        _, C, H, W = dummy_latents.shape

    # 3) for each severity, run the pipe
    rows = []
    for sev in severities:
        ctrl_map = make_severity_map(sev, H, W, device)
        out = pipe(
            prompt_embeds              = txt_embeds,
            controlnet_conditioning_image = ctrl_map,
            num_inference_steps        = 50,
            guidance_scale             = 7.5,
        ).images  # list of PIL images

        # convert each PIL→Tensor in [0,1]
        toks = [
            torch.from_numpy(np.array(im).transpose(2,0,1) / 255.0)
            for im in out
        ]
        rows.append(torch.stack(toks, dim=0))

    # 4) make a grid, plot & save
    grid = make_grid(torch.cat(rows, dim=0), nrow=batch)
    plt.figure(figsize=(batch*3, len(severities)*3))
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.axis("off")
    plt.title("Rows = severities, Cols = samples")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved visualization to {out_path}")


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      type=str, required=True,
                        help="path to your fine-tuned ControlNet .pth")
    parser.add_argument("--data_root", type=str, required=True,
                        help="root of your dataset (must contain `labels/` & `images/` sub-dirs)")
    parser.add_argument("--max_samples", type=int, default=4,
                        help="how many examples to visualize")
    parser.add_argument("--out",       type=str, default="results.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe   = load_pipeline(args.ckpt, device)

    # ← use the recursive version
    ds = XBDFullDataset(
        labels_root = os.path.join(args.data_root, "labels"),
        images_root = os.path.join(args.data_root, "images"),
        crop_size   = 512,
        max_samples = args.max_samples,
        annotate    = False,
    )

    # gather your pre-disaster crops
    pre_imgs   = [ ds[i]["pre"] for i in range(min(len(ds), args.max_samples)) ]
    # severities you want to sweep through
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]

    infer_and_plot(pipe, pre_imgs, severities, out_path=args.out)
    print("Done!") 
