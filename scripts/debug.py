#!/usr/bin/env python3
"""
Debug script to inspect dataset globs, JSON pairing, and environment variables
for XBDPairDataset (recursive) across multiple roots — now counting both
pre- and post-disaster images.
"""
import os
import glob
import logging
from pathlib import Path
from src.datasets import XBDFullDataset

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")


def debug_root(root, crop_size, max_samples=None, annotate=False):
    labels_root = Path(root) / "labels"
    images_root = Path(root) / "images"

    logging.info(f"--- ROOT: {root} ---")
    logging.info(f" labels_root exists? {labels_root.exists()}")
    logging.info(f" images_root exists? {images_root.exists()}")

    # 1) count post-disaster PNG and JSON
    post_pngs  = sorted(glob.glob(str(images_root / "**" / "*_post_disaster.png"), recursive=True))
    post_jsons = sorted(glob.glob(str(labels_root / "**" / "*_post_disaster.json"), recursive=True))
    logging.info(f" Found {len(post_pngs)} *_post_disaster.png files")
    logging.info(f" Found {len(post_jsons)} *_post_disaster.json files")

    # 2) count pre-disaster PNG
    pre_pngs = sorted(glob.glob(str(images_root / "**" / "*_pre_disaster.png"), recursive=True))
    logging.info(f" Found {len(pre_pngs)} *_pre_disaster.png files")

    # 3) sample missing JSON for first 1k posts
    missing = []
    for p in post_pngs[:1000]:
        rel = os.path.relpath(p, images_root)
        j   = labels_root / rel.replace(".png", ".json")
        if not j.exists():
            missing.append(rel)
    logging.info(f" Of first {min(1000,len(post_pngs))} post-PNGs, {len(missing)} missing JSON (show up to 10):")
    for r in missing[:10]:
        logging.info(f"   MISSING JSON → {r}")

    # 4) instantiate and peek the paired dataset
    ds = XBDFullDataset(
        labels_root=str(labels_root),
        images_root=str(images_root),
        crop_size=crop_size,
        max_samples=max_samples,
        annotate=annotate,
    )
    logging.info(f" XBDFullDataset (i.e. valid post+pre pairs) length = {len(ds)}")

    # 5) try to load a couple of samples
    for idx in range(min(5, len(ds))):
        try:
            item = ds[idx]
            logging.info(
                f"  idx={idx}: pre={item['pre'].shape}, post={item['post'].shape}, "
                f"mask.sum={item['mask'].sum():.4f}, severity={item['severity']:.4f}"
            )
        except Exception as e:
            logging.error(f"  idx={idx}: failed to load pair — {e!r}")

    return len(ds)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_roots", type=str, required=True,
                   help="Comma-separated roots: data/train,data/tier3,data/test")
    p.add_argument("--crop_size",  type=int, default=512)
    p.add_argument("--max_samples",type=int, default=None)
    p.add_argument("--annotate",   action="store_true")
    args = p.parse_args()

    logging.info(
        f"ENV VARS: RANK={os.environ.get('RANK')}, "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}"
    )

    roots = [r.strip() for r in args.data_roots.split(",") if r.strip()]
    total = 0
    for root in roots:
        total += debug_root(root, args.crop_size, args.max_samples, args.annotate)
    logging.info(f"=== GRAND TOTAL of valid pairs across {roots}: {total} ===")


if __name__ == "__main__":
    main()
