# src/datasets/xbd_pair.py
import os
import json
from pathlib import Path
from PIL import Image, ImageDraw
from shapely import wkt
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as T

damage_colors = {
    "no-damage":     (0, 255,   0, 50),
    "minor-damage":  (0,   0, 255, 50),
    "major-damage":  (255, 69,   0, 50),
    "destroyed":     (255,   0,   0, 50),
    "un-classified": (255, 255, 255, 50),
}

def _load_json(path):
    # try utf-8, fallback to latin-1
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Could not decode JSON: {path}")

class XBDFullDataset(Dataset):
    """
    Recursively collects all *_post_disaster.json under labels_root,
    pairs with their pre/post PNGs under images_root, crops and normalizes.
    """

    def __init__(
        self,
        labels_root: str,
        images_root: str,
        crop_size: int = 512,
        max_samples: int = None,
        annotate: bool = False,
    ):
        self.labels_root = Path(labels_root)
        self.images_root = Path(images_root)
        self.crop_size = crop_size
        self.annotate = annotate
        self.normalize = T.Normalize([0.5]*3, [0.5]*3)

        # build list of (json, post_png, pre_png)
        self.items = []
        for post_json in self.labels_root.rglob("*_post_disaster.json"):
            rel = post_json.relative_to(self.labels_root)
            post_png = (self.images_root / rel).with_suffix(".png")
            pre_png = post_png.with_name(
                post_png.name.replace("_post_disaster.png", "_pre_disaster.png")
            )
            if post_png.exists() and pre_png.exists():
                self.items.append((str(post_json), str(post_png), str(pre_png)))
        if max_samples:
            self.items = self.items[:max_samples]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        post_json, post_png, pre_png = self.items[idx]

        # 1) load metadata + features, with encoding fallback
        data = _load_json(post_json)
        meta = data.get("metadata", {})
        feats = data.get("features", {})
        coords = feats.get("xy", []) if isinstance(feats, dict) else []

        # 2) load images
        post_img = Image.open(post_png).convert("RGB")
        pre_img = Image.open(pre_png).convert("RGB")

        # 3) rasterize mask
        mask = Image.new("L", post_img.size, 0)
        draw = ImageDraw.Draw(mask)
        for feat in coords:
            poly = wkt.loads(feat["wkt"])
            pts = list(zip(poly.exterior.coords.xy[0], poly.exterior.coords.xy[1]))
            draw.polygon(pts, fill=1)
        del draw

        # 4) optional annotation overlay
        if self.annotate:
            ad = ImageDraw.Draw(pre_img, "RGBA")
            bd = ImageDraw.Draw(post_img, "RGBA")
            for feat in coords:
                sub = feat.get("properties", {}).get("subtype", "no-damage")
                col = damage_colors.get(sub, damage_colors["no-damage"])
                poly = wkt.loads(feat["wkt"])
                pts = list(zip(poly.exterior.coords.xy[0], poly.exterior.coords.xy[1]))
                ad.polygon(pts, col)
                bd.polygon(pts, col)
            del ad, bd

        # 5) joint random crop
        i, j, h, w = T.RandomCrop.get_params(
            pre_img, (self.crop_size, self.crop_size)
        )
        pre_crop = F.crop(pre_img, i, j, h, w)
        post_crop = F.crop(post_img, i, j, h, w)
        mask_crop = F.crop(mask, i, j, h, w)

        # 6) to-tensor + normalize
        pre_t = self.normalize(F.to_tensor(pre_crop))
        post_t = self.normalize(F.to_tensor(post_crop))
        mask_t = F.to_tensor(mask_crop)

        # 7) severity
        severity = mask_t.mean()

        return {
            "pre": pre_t,          # [3,H,W]
            "post": post_t,        # [3,H,W]
            "mask": mask_t,        # [1,H,W]
            "severity": severity,  # scalar
            "meta": meta,          # dict
        }
