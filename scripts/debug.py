#!/usr/bin/env python3
"""
Debug script to inspect dataset globs, JSON pairing, and environment variables
for XBDPairDataset across multiple roots.
"""
import os
import glob
import json
import logging
from pathlib import Path
from src.datasets import XBDPairDataset

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")

def debug_root(root, crop_size, max_samples=None, annotate=False):
    labels_dir = Path(root) / "labels"
    images_dir = Path(root) / "images"

    logging.info(f"--- ROOT: {root} ---")
    logging.info(f" labels_dir exists? {labels_dir.exists()}")
    logging.info(f" images_dir exists? {images_dir.exists()}")

    # count PNGs and JSONs
    pngs = sorted(glob.glob(str(images_dir / "**" / "*_post_disaster.png"), recursive=True))
    jsons = sorted(glob.glob(str(labels_dir / "**" / "*_post_disaster.json"), recursive=True))
    logging.info(f" Found {len(pngs)} post_disaster.png files")
    logging.info(f" Found {len(jsons)} post_disaster.json files")

    # look for PNGs that have no matching JSON
    missing_json = []
    for p in pngs[:1000]:  # sample first 1k for speed
        rel = os.path.relpath(p, images_dir)
        j = labels_dir / rel.replace(".png", ".json")
        if not j.exists():
            missing_json.append(rel)
    logging.info(f" Of first {min(1000,len(pngs))} PNGs, {len(missing_json)} have no JSON (showing up to 10):")
    for rel in missing_json[:10]:
        logging.info(f"   MISSING JSON for {rel}")

    # instantiate dataset and peek first few __getitem__
    ds = XBDPairDataset(
        labels_dir=str(labels_dir),
        images_dir=str(images_dir),
        crop_size=crop_size,
        max_samples=max_samples,
        annotate=annotate,
    )
    logging.info(f" XBDPairDataset reports length={len(ds)}")
    for idx in range(min(5, len(ds))):
        item = ds[idx]
        logging.info(f"  idx={idx}: pre.shape={item['pre'].shape}, post.shape={item['post'].shape}, "
                     f"mask.sum={item['mask'].sum():.4f}, severity={item['severity']:.4f}")
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

    # log environment
    logging.info(f"ENV VARS: RANK={os.environ.get('RANK')}, "
                 f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
                 f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}")

    roots = [r.strip() for r in args.data_roots.split(",") if r.strip()]
    total = 0
    for root in roots:
        cnt = debug_root(root, args.crop_size, args.max_samples, args.annotate)
        total += cnt
    logging.info(f"=== TOTAL across {roots}: {total} ===")

if __name__ == "__main__":
    main()
