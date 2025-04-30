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
        # Recursively collect all post‐disaster JSONs
        pattern = os.path.join(labels_dir, "**", "*_post_disaster.json")
        self.post_jsons = sorted(glob.glob(pattern, recursive=True))
        if max_samples:
            self.post_jsons = self.post_jsons[:max_samples]

        self.images_dir = images_dir
        self.crop_size  = crop_size
        self.annotate   = annotate
        self.normalize  = T.Normalize([0.5]*3, [0.5]*3)

    def __len__(self):
        return len(self.post_jsons)

    def __getitem__(self, idx):
        post_json = self.post_jsons[idx]
        pre_json  = post_json.replace("_post_disaster.json", "_pre_disaster.json")

        post_data = json.load(open(post_json))
        meta_post = post_data["metadata"]

        # compute relative path under "labels/**"
        rel       = os.path.relpath(post_json, os.path.join(*post_json.split(os.sep)[:2], "labels"))
        png_rel   = rel.replace(".json", ".png")
        post_png  = os.path.join(self.images_dir, png_rel)
        pre_png   = post_png.replace("_post_disaster.png", "_pre_disaster.png")

        post_img = Image.open(post_png).convert("RGB")
        pre_img  = Image.open(pre_png).convert("RGB")

        # build binary mask
        features = post_data.get("features", {}).get("xy", [])
        mask = Image.new("L", post_img.size, 0)
        draw = ImageDraw.Draw(mask)
        for feat in features:
            poly   = wkt.loads(feat["wkt"])
            coords = list(zip(poly.exterior.coords.xy[0], poly.exterior.coords.xy[1]))
            draw.polygon(coords, fill=1)
        del draw

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

        # joint random crop
        i, j, h, w = T.RandomCrop.get_params(pre_img, (self.crop_size, self.crop_size))
        pre_crop  = F.crop(pre_img,  i, j, h, w)
        post_crop = F.crop(post_img, i, j, h, w)
        mask_crop = F.crop(mask,    i, j, h, w)

        pre_t  = self.normalize(F.to_tensor(pre_crop))
        post_t = self.normalize(F.to_tensor(post_crop))
        mask_t = F.to_tensor(mask_crop)

        severity = mask_t.mean()

        return {
            "pre":      pre_t,
            "post":     post_t,
            "mask":     mask_t,
            "severity": severity,
            "meta":     meta_post,
        }
