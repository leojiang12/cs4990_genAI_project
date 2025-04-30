# src/datasets/xbd_pair.py (or wherever your XBDPairDataset lives)
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
                 labels_dir,     # e.g. "data/train/labels"
                 images_dir,     # e.g. "data/train/images"
                 crop_size=512,
                 max_samples=None,
                 annotate=False):
        # 1) find every post‐disaster PNG under images_dir
        pattern = os.path.join(images_dir, "**", "*_post_disaster.png")
        self.post_pngs = sorted(glob.glob(pattern, recursive=True))
        if max_samples:
            self.post_pngs = self.post_pngs[:max_samples]

        self.labels_dir = labels_dir
        self.images_dir = images_dir
        self.crop_size  = crop_size
        self.annotate   = annotate
        self.normalize  = T.Normalize([0.5]*3, [0.5]*3)

    def __len__(self):
        return len(self.post_pngs)

    def __getitem__(self, idx):
        # 2) get the full-res post‐disaster PNG
        post_png = self.post_pngs[idx]
        # 3) derive the matching JSON (might not exist!)
        rel_path = os.path.relpath(post_png, self.images_dir)
        post_json = os.path.join(self.labels_dir,
                                 rel_path.replace(".png", ".json"))

        # try to load JSON; if missing, pretend “no features”
        if os.path.exists(post_json):
            post_data = json.load(open(post_json))
            features = post_data.get("features", {}).get("xy", [])
            meta_post = post_data.get("metadata", {})
        else:
            features = []
            meta_post = {}

        # 4) load pre/post images
        pre_png  = post_png.replace("_post_disaster.png", "_pre_disaster.png")
        post_img = Image.open(post_png).convert("RGB")
        pre_img  = Image.open(pre_png).convert("RGB")

        # 5) rasterize polygons (if any) into a mask
        mask = Image.new("L", post_img.size, 0)
        draw = ImageDraw.Draw(mask)
        for feat in features:
            poly   = wkt.loads(feat["wkt"])
            coords = list(zip(poly.exterior.coords.xy[0],
                              poly.exterior.coords.xy[1]))
            draw.polygon(coords, fill=1)
        del draw

        # 6) optional annotation overlay
        if self.annotate:
            ad = ImageDraw.Draw(pre_img,  "RGBA")
            bd = ImageDraw.Draw(post_img, "RGBA")
            for feat in features:
                sub   = feat["properties"].get("subtype", "no-damage")
                color = damage_colors.get(sub, damage_colors["no-damage"])
                poly   = wkt.loads(feat["wkt"])
                coords = list(zip(poly.exterior.coords.xy[0],
                                  poly.exterior.coords.xy[1]))
                ad.polygon(coords, color)
                bd.polygon(coords, color)
            del ad, bd

        # 7) joint random crop
        i, j, h, w = T.RandomCrop.get_params(pre_img,
                                             output_size=(self.crop_size, self.crop_size))
        pre_crop  = F.crop(pre_img,  i, j, h, w)
        post_crop = F.crop(post_img, i, j, h, w)
        mask_crop = F.crop(mask,    i, j, h, w)

        # 8) to‐tensor & normalize
        pre_t  = self.normalize(F.to_tensor(pre_crop))
        post_t = self.normalize(F.to_tensor(post_crop))
        mask_t = F.to_tensor(mask_crop)

        # 9) severity = fraction of damaged pixels
        severity = mask_t.mean()

        return {
            "pre":      pre_t,       # [3,H,W]
            "post":     post_t,      # [3,H,W]
            "mask":     mask_t,      # [1,H,W]
            "severity": severity,    # scalar in [0,1]
            "meta":     meta_post,   # metadata if available, else {}
        }
