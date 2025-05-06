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
