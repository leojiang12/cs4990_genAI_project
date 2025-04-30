# scripts/infer_and_visualize_fixed.py

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import CLIPTokenizer

from src.datasets import XBDPairDataset

def load_pipeline(controlnet_ckpt, device="cuda"):
    # Load the full ControlNet pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth"),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)

    # Overwrite the ControlNet weights with your checkpoint:
    pipe.controlnet.load_state_dict(torch.load(controlnet_ckpt, map_location=device))

    # Enable attention slicing to reduce VRAM (optional, but safe on a laptop GPU)
    pipe.enable_attention_slicing()

    return pipe

def make_severity_map(severity: float, size: int) -> Image.Image:
    """
    Return a constant‐gray RGB PIL image of shape (size×size)
    where each pixel = int(severity * 255).
    """
    val = int(severity * 255)
    arr = np.full((size, size, 3), val, dtype=np.uint8)
    return Image.fromarray(arr)

def infer_and_plot(pipe, pre_imgs, severities, out_path="results.png"):
    device = pipe.device
    batch = len(pre_imgs)

    # Convert pre-images to PIL (pipeline expects PIL or tensor in [0,1])
    # Our dataset gives us normalized tensors [-1,1], so we unnormalize:
    pil_pre = []
    for t in pre_imgs:
        img = ((t * 0.5 + 0.5) * 255).clamp(0,255).cpu().numpy().astype(np.uint8)
        img = img.transpose(1,2,0)
        pil_pre.append(Image.fromarray(img))

    # We’ll use the same prompt for all
    prompts = ["photo"] * batch

    rows = []
    for sev in severities:
        ctrl_maps = [make_severity_map(sev, pipe.vae.config.sample_size, 
                                       pipe.vae.config.sample_size, device)
                    for _ in range(batch)]

        # Run the pipeline
        outputs = pipe(
            prompt=prompts,
            image=ctrl_maps, 
            controlnet_conditioning_image=ctrl_maps,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images  # this is a list of PIL

        # Convert to tensor for grid
        toks = []
        for im in outputs:
            arr = np.array(im).astype(np.float32)/255.0
            toks.append(torch.from_numpy(arr).permute(2,0,1))
        rows.append(torch.stack(toks, dim=0))

    # Build a (rows × batch) grid
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
    parser.add_argument("--ckpt",      required=True,
                        help="Path to ControlNet .pth checkpoint")
    parser.add_argument("--data_root", required=True,
                        help="Root containing `labels/` and `images/`")
    parser.add_argument("--out",       default="severity_sweep.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe   = load_pipeline(args.ckpt, device)

    # Load 4 samples from your holdout
    ds = XBDPairDataset(
        labels_dir=os.path.join(args.data_root,"labels"),
        images_dir=os.path.join(args.data_root,"images"),
        crop_size=512, max_samples=4, annotate=False
    )
    if len(ds)==0:
        raise RuntimeError(f"No samples in {args.data_root}/labels or images")

    pre_imgs   = [ ds[i]["pre"] for i in range(min(len(ds),4)) ]
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]

    infer_and_plot(pipe, pre_imgs, severities, out_path=args.out)
