# scripts/infer_and_visualize.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
)
from transformers import CLIPTextModel, CLIPTokenizer

from src.datasets import XBDPairDataset

def load_pipeline(controlnet_ckpt, device="cuda"):
    # 1) Base SD components
    vae      = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
    tokenizer= CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_enc = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    unet     = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(device)

    # 2) ControlNet (depth template) then load your fine-tuned weights
    ctrl_net = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth").to(device)
    ctrl_net.load_state_dict(torch.load(controlnet_ckpt, map_location=device))
    ctrl_net.requires_grad_(False)

    # 3) Build the pipeline
    pipe = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_enc, tokenizer=tokenizer,
        unet=unet, controlnet=ctrl_net,
        scheduler=pipe.scheduler, safety_checker=None, feature_extractor=None
    ).to(device)
    pipe.enable_model_cpu_offload()  # optional, for memory
    return pipe

def make_severity_map(severity, H, W, device):
    # severity: float in [0,1]
    m = torch.full((1, 1, H, W), float(severity), device=device)
    return m

def infer_and_plot(pipe, pre_imgs, severities, out_path="results.png"):
    device = pipe.device
    batch = len(pre_imgs)
    # encode pre-images to latents once
    with torch.no_grad():
        pre_tensors = torch.stack([img.to(device) for img in pre_imgs], dim=0)
        # dummy prompt “photo”
        prompts = ["photo"] * batch
        text_inputs = pipe.tokenizer(prompts, return_tensors="pt",
                                     padding="max_length", truncation=True,
                                     max_length=pipe.tokenizer.model_max_length).to(device)
        txt_embeds = pipe.text_encoder(**text_inputs).last_hidden_state

        # get post-latents shape from vae (for scheduling)
        # here we just run encode on a dummy to infer shape
        dummy_post = pre_tensors
        post_latents = pipe.vae.encode(dummy_post).latent_dist.sample() * pipe.vae.config.scaling_factor
        _, C, H, W = post_latents.shape

    rows = []
    for sev in severities:
        # build control map
        ctrl_map = make_severity_map(sev, H, W, device)
        # run diffusion
        images = pipe(
            prompt_embeds=txt_embeds,
            controlnet_conditioning_image=ctrl_map,
            num_inference_steps=50,
            guidance_scale=7.5,
        ).images  # list of PIL

        # convert to tensor grid
        toks = [torch.from_numpy(np.array(im).transpose(2,0,1)/255.0) for im in images]
        rows.append(torch.stack(toks, dim=0))

    # stack rows [len(severities), batch, 3, H, W]
    grid = make_grid(torch.cat(rows, dim=0), nrow=batch)
    plt.figure(figsize=(batch*3, len(severities)*3))
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.axis("off")
    plt.title("Rows = severities, Cols = samples")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to your ControlNet checkpoint")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out", type=str, default="results.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipeline(args.ckpt, device)

    # load a few validation samples
    ds = XBDPairDataset(
        labels_dir=os.path.join(args.data_root,"labels_val"),
        images_dir=os.path.join(args.data_root,"images_val"),
        crop_size=512, max_samples=4, annotate=False
    )
    # pull out first 4 pre-images
    pre_imgs = [ds[i]["pre"] for i in range(min(len(ds),4))]

    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
