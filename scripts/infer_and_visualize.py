#!/usr/bin/env python3
import os
import sys
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


def infer_and_plot(pipe, pre_imgs, masks, out_path="severity_sweep.png"):
    """
    pre_imgs: list of [3×H×W] tensors in [-1,1]
    masks:    list of [1×H×W] binary tensors {0,1}
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    from PIL import Image

    device = pipe.device
    B      = len(pre_imgs)
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]

    # 1) Convert pre_imgs → PIL and keep as tensors for grid
    pil_pre, toks_pre = [], []
    for t in pre_imgs:
        arr = ((t * 0.5 + 0.5) * 255).clamp(0,255).byte().cpu().numpy()
        arr = arr.transpose(1,2,0)
        pil_pre.append(Image.fromarray(arr))
        toks_pre.append(torch.from_numpy(arr.transpose(2,0,1)/255.0))

    # 2) Prepare text embeddings once
    prompts = ["photo"] * B
    tokens  = pipe.tokenizer(prompts, return_tensors="pt",
                              padding="max_length", truncation=True,
                              max_length=pipe.tokenizer.model_max_length).to(device)
    txt_emb = pipe.text_encoder(**tokens).last_hidden_state

    all_rows = []
    for i in range(B):
        row_toks = [toks_pre[i]]  # start with the pre‑disaster tensor
        # for each severity level, build a gray mask and run the pipeline
        for sev in severities:
            # scale the binary mask by sev → float mask
            m = masks[i].float().to(device)            # [1,H,W] in {0,1}
            sev_mask = (m * sev * 255).byte().cpu().numpy().squeeze(0)
            rgb = np.stack([sev_mask]*3, axis=-1)      # [H,W,3] gray
            pil_mask = Image.fromarray(rgb)

            out = pipe(
                prompt_embeds=txt_emb[i : i+1],
                image=[pil_pre[i]],
                controlnet_conditioning_image=[pil_mask],
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]

            arr = np.array(out)
            row_toks.append(torch.from_numpy(arr.transpose(2,0,1)/255.0))

        all_rows.extend(row_toks)

    # 3) Make a big grid: B rows, 1+5 columns
    grid = make_grid(torch.stack(all_rows, dim=0), nrow=1 + len(severities), pad_value=1.0)

    plt.figure(figsize=((1+len(severities)) * 3, B * 3))
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved severity sweep to {out_path}")


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       required=True,
                        help="path to ControlNet .pth checkpoint")
    parser.add_argument("--data_root",  required=True,
                        help="root of your dataset (labels/ & images/)")
    parser.add_argument("--max_samples",type=int, default=4,
                        help="how many examples to visualize")
    parser.add_argument("--out",        default="results.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe   = load_pipeline(args.ckpt, device)

    ds = XBDFullDataset(
        labels_root = os.path.join(args.data_root,"labels"),
        images_root = os.path.join(args.data_root,"images"),
        crop_size   = 512,
        max_samples = args.max_samples,
        annotate    = False,
    )
    if len(ds)==0:
        raise RuntimeError("No samples found")

    pre_imgs = [ ds[i]["pre"]  for i in range(len(ds)) ]
    masks    = [ ds[i]["mask"] for i in range(len(ds)) ]
    infer_and_plot(pipe, pre_imgs, masks, out_path=args.out)
