# src/datasets/xbd_pair.py
import os, glob, json
from PIL import Image, ImageDraw
from shapely import wkt
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as T

damage_colors = {
    "no-damage":     (0, 255,   0, 50),
    "minor-damage":  (0,   0, 255, 50),
    "major-damage":  (255, 69,   0, 50),
    "destroyed":     (255,   0,   0, 50),
    "un-classified": (255,255, 255, 50),
}

class XBDPairDataset(Dataset):
    def __init__(self,
                 labels_dir: str,    # e.g. "data/train/labels"
                 images_dir: str,    # e.g. "data/train/images"
                 crop_size: int = 512,
                 max_samples: int = None,
                 annotate: bool = False):
        self.labels_dir = Path(labels_dir)
        self.images_dir = Path(images_dir)
        self.crop_size  = crop_size
        self.annotate   = annotate

        # 1) glob all post-disaster JSONs recursively
        pattern = str(self.labels_dir / "**" / "*_post_disaster.json")
        self.post_jsons = sorted(glob.glob(pattern, recursive=True))
        if max_samples:
            self.post_jsons = self.post_jsons[:max_samples]

        # normalization to [-1,1]
        self.normalize = T.Normalize([0.5]*3, [0.5]*3)

    def __len__(self):
        return len(self.post_jsons)

    def __getitem__(self, idx):
        post_json = self.post_jsons[idx]

        # helper to load JSON with fallback encoding
        def load_json(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except UnicodeDecodeError:
                with open(path, 'r', encoding='latin-1') as f:
                    return json.load(f)

        # 2) derive matching PNG paths
        rel = Path(post_json).relative_to(self.labels_dir)
        post_png = self.images_dir / rel.with_suffix(".png").name.replace(
            "_post_disaster.json", "_post_disaster.png"
        )
        pre_png = Path(str(post_png)).with_name(
            post_png.name.replace("_post_disaster.png", "_pre_disaster.png")
        )

        # 3) load metadata + features
        data = load_json(post_json)
        feats = data.get("features", {}).get("xy", [])
        meta  = data.get("metadata", {})

        # 4) open images
        post_img = Image.open(post_png).convert("RGB")
        pre_img  = Image.open(pre_png).convert("RGB")

        # 5) rasterize polygons → mask
        mask = Image.new("L", post_img.size, 0)
        draw = ImageDraw.Draw(mask)
        for f in feats:
            poly = wkt.loads(f["wkt"])
            coords = list(zip(poly.exterior.coords.xy[0], poly.exterior.coords.xy[1]))
            draw.polygon(coords, fill=1)
        del draw

        # 6) optional annotation overlay
        if self.annotate:
            ad = ImageDraw.Draw(pre_img,  "RGBA")
            bd = ImageDraw.Draw(post_img, "RGBA")
            for f in feats:
                sub   = f.get("properties", {}).get("subtype", "no-damage")
                col   = damage_colors.get(sub, damage_colors["no-damage"])
                poly  = wkt.loads(f["wkt"])
                coords= list(zip(poly.exterior.coords.xy[0], poly.exterior.coords.xy[1]))
                ad.polygon(coords, col)
                bd.polygon(coords, col)
            del ad, bd

        # 7) joint random crop
        i, j, h, w = T.RandomCrop.get_params(pre_img, (self.crop_size, self.crop_size))
        pre_c  = F.crop(pre_img,  i, j, h, w)
        post_c = F.crop(post_img, i, j, h, w)
        mask_c = F.crop(mask,    i, j, h, w)

        # 8) to-tensor & normalize
        pre_t  = self.normalize(F.to_tensor(pre_c))
        post_t = self.normalize(F.to_tensor(post_c))
        mask_t = F.to_tensor(mask_c)

        # 9) severity = fraction of damaged pixels
        severity = mask_t.mean()

        return {
            "pre":      pre_t,       # [3,H,W]
            "post":     post_t,      # [3,H,W]
            "mask":     mask_t,      # [1,H,W]
            "severity": severity,    # scalar in [0,1]
            "meta":     meta,        # metadata if any
        }

# src/datasets/xbd_pair.py
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

def _load_json(path: Path):
    # Try UTF-8 first, then Latin-1 if that fails
    for enc in ("utf-8", "latin-1"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Could not decode JSON at {path!r}")

class XBDFullDataset(Dataset):
    """
    Recursively collects all *_post_disaster.json under labels_root,
    pairs them with pre/post PNGs under images_root, optionally
    crops, normalizes, and annotates.
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
        self.crop_size   = crop_size
        self.annotate    = annotate
        self.normalize   = T.Normalize([0.5]*3, [0.5]*3)

        # Build (json, post_png, pre_png) list
        self.items = []
        for post_json in self.labels_root.rglob("*_post_disaster.json"):
            rel     = post_json.relative_to(self.labels_root)
            post_png = (self.images_root / rel).with_suffix(".png")
            pre_png  = post_png.with_name(
                post_png.name.replace("_post_disaster.png", "_pre_disaster.png")
            )
            if post_png.exists() and pre_png.exists():
                self.items.append((post_json, post_png, pre_png))
        if max_samples:
            self.items = self.items[:max_samples]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        post_json, post_png, pre_png = self.items[idx]

        # 1) load metadata + features
        data   = _load_json(post_json)
        meta   = data.get("metadata", {})
        feats  = data.get("features", {}).get("xy", [])

        # 2) open images
        post_img = Image.open(post_png).convert("RGB")
        pre_img  = Image.open(pre_png).convert("RGB")

        # 3) rasterize mask
        mask = Image.new("L", post_img.size, 0)
        draw = ImageDraw.Draw(mask)
        for feat in feats:
            poly   = wkt.loads(feat["wkt"])
            pts    = list(zip(poly.exterior.coords.xy[0], poly.exterior.coords.xy[1]))
            draw.polygon(pts, fill=1)
        del draw

        # 4) optional annotation
        if self.annotate:
            ad = ImageDraw.Draw(pre_img,  "RGBA")
            bd = ImageDraw.Draw(post_img, "RGBA")
            for feat in feats:
                sub   = feat.get("properties", {}).get("subtype", "no-damage")
                col   = damage_colors.get(sub, damage_colors["no-damage"])
                poly  = wkt.loads(feat["wkt"])
                pts   = list(zip(poly.exterior.coords.xy[0], poly.exterior.coords.xy[1]))
                ad.polygon(pts, col)
                bd.polygon(pts, col)
            del ad, bd

        # 5) joint random crop
        i, j, h, w = T.RandomCrop.get_params(
            pre_img, (self.crop_size, self.crop_size)
        )
        pre_crop  = F.crop(pre_img,  i, j, h, w)
        post_crop = F.crop(post_img, i, j, h, w)
        mask_crop = F.crop(mask,    i, j, h, w)

        # 6) to-tensor + normalize
        pre_t  = self.normalize(F.to_tensor(pre_crop))
        post_t = self.normalize(F.to_tensor(post_crop))
        mask_t = F.to_tensor(mask_crop)

        # 7) severity
        severity = mask_t.mean()

        return {
            "pre":      pre_t,       # [3,H,W]
            "post":     post_t,      # [3,H,W]
            "mask":     mask_t,      # [1,H,W]
            "severity": severity,    # scalar
            "meta":     meta,        # metadata dict
        }
