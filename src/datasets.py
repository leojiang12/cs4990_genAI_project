import os, glob, json
from PIL import Image, ImageDraw
from shapely import wkt
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as T
import random

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
        # only pick the post‐disaster JSONs
        self.post_jsons = sorted(glob.glob(os.path.join(labels_dir, "*_post_disaster.json")))
        if max_samples:
            self.post_jsons = self.post_jsons[:max_samples]

        self.images_dir = images_dir
        self.crop_size  = crop_size
        self.annotate   = annotate

        # normalization to [-1,1]
        self.normalize = T.Normalize([0.5]*3, [0.5]*3)

    def __len__(self):
        return len(self.post_jsons)

    def __getitem__(self, idx):
        post_json = self.post_jsons[idx]
        pre_json  = post_json.replace("_post_disaster.json", "_pre_disaster.json")

        # --- load JSON metadata ---
        post_data = json.load(open(post_json))
        pre_data  = json.load(open(pre_json))
        meta_post = post_data["metadata"]

        # --- derive PNG paths ---
        # labels/.../joplin-tornado_..._post_disaster.json
        # -> images/.../joplin-tornado_..._post_disaster.png
        rel = os.path.relpath(post_json, start=os.path.dirname(self.post_jsons[0]))
        png_rel = rel.replace("labels", "").replace(".json", ".png")
        post_png = os.path.join(self.images_dir, png_rel)
        pre_png  = post_png.replace("_post_disaster.png", "_pre_disaster.png")

        # --- open PIL images ---
        post_img = Image.open(post_png).convert("RGB")
        pre_img  = Image.open(pre_png).convert("RGB")

        # # --- make binary mask from post polygons ---
        # mask = Image.new("L", post_img.size, 0)
        # draw = ImageDraw.Draw(mask)
        # for feat in post_data.get("features", []):
        #     poly = wkt.loads(feat["wkt"])
        #     coords = list(zip(poly.exterior.coords.xy[0], poly.exterior.coords.xy[1]))
        #     draw.polygon(coords, fill=1)
        # del draw
        # --- extract the actual polygon list under "features" → "xy" ---
        feat_dict = post_data.get("features", {})
        features  = feat_dict.get("xy", []) if isinstance(feat_dict, dict) else []
        # --- rasterize those polygons into a binary mask ---
        mask = Image.new("L", post_img.size, 0)
        draw = ImageDraw.Draw(mask)
        for feat in features:
            poly   = wkt.loads(feat["wkt"])
            coords = list(zip(poly.exterior.coords.xy[0], poly.exterior.coords.xy[1]))
            draw.polygon(coords, fill=1)
        del draw

        # # --- optionally annotate RGB with translucent polygons ---
        # if self.annotate:
        #     ad = ImageDraw.Draw(pre_img, "RGBA")
        #     bd = ImageDraw.Draw(post_img, "RGBA")
        #     for feat in post_data.get("features", []):
        #         color = damage_colors[feat["properties"].get("subtype","no-damage")]
        #         poly = wkt.loads(feat["wkt"])
        #         coords = list(zip(poly.exterior.coords.xy[0], poly.exterior.coords.xy[1]))
        #         ad.polygon(coords, color)
        #         bd.polygon(coords, color)
        #     del ad, bd
        # --- optionally annotate RGB with translucent polygons ---
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

        # --- joint random crop ---
        i, j, h, w = T.RandomCrop.get_params(pre_img, output_size=(self.crop_size, self.crop_size))
        pre_crop  = F.crop(pre_img,  i, j, h, w)
        post_crop = F.crop(post_img, i, j, h, w)
        mask_crop = F.crop(mask,    i, j, h, w)

        # --- to tensor & normalize ---
        pre_t  = self.normalize(F.to_tensor(pre_crop))
        post_t = self.normalize(F.to_tensor(post_crop))
        mask_t = F.to_tensor(mask_crop)

        # compute severity = fraction of damaged pixels
        severity = mask_t.mean()  # a scalar tensor in [0,1]

        return {
            "pre":  pre_t,       # [3,H,W]
            "post": post_t,      # [3,H,W]
            "mask": mask_t,      # [1,H,W]
            "severity": severity,     # torch.Tensor(1)
            "meta": meta_post,   # raw metadata for post image
        }
