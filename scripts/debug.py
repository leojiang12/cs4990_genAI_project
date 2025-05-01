#!/usr/bin/env python3
"""
Debug script to inspect dataset globs, JSON pairing, and environment variables
for XBDPairDataset (recursive) across multiple roots.
"""
import os
import glob
import json
import logging
from pathlib import Path
from src.datasets import XBDFullDataset  # ← your recursive version

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")


def debug_root(root, crop_size, max_samples=None, annotate=False):
    labels_root = Path(root) / "labels"
    images_root = Path(root) / "images"

    logging.info(f"--- ROOT: {root} ---")
    logging.info(f" labels_root exists? {labels_root.exists()}")
    logging.info(f" images_root exists? {images_root.exists()}")

    # recursive find
    pngs = sorted(glob.glob(str(images_root / "**" / "*_post_disaster.png"), recursive=True))
    jsons = sorted(glob.glob(str(labels_root / "**" / "*_post_disaster.json"), recursive=True))
    logging.info(f" Found {len(pngs)} post_disaster.png files")
    logging.info(f" Found {len(jsons)} post_disaster.json files")

    # sample missing JSON
    missing = []
    for p in pngs[:1000]:
        rel = os.path.relpath(p, images_root)
        j   = labels_root / rel.replace(".png", ".json")
        if not j.exists():
            missing.append(rel)
    logging.info(f" Of first {min(1000,len(pngs))} PNGs, {len(missing)} missing JSON (show up to 10):")
    for r in missing[:10]:
        logging.info(f"   MISSING JSON → {r}")

    # instantiate and peek
    ds = XBDFullDataset(
        labels_root=str(labels_root),
        images_root=str(images_root),
        crop_size=crop_size,
        max_samples=max_samples,
        annotate=annotate,
    )
    logging.info(f" XBDFullDataset length = {len(ds)}")

    # try to read first few items, catching JSON errors
    good = 0
    for idx in range(min(5, len(ds))):
        try:
            item = ds[idx]
            logging.info(
                f"  idx={idx}: pre.shape={item['pre'].shape}, "
                f"post.shape={item['post'].shape}, "
                f"mask.sum={item['mask'].sum():.4f}, "
                f"severity={item['severity']:.4f}"
            )
            good += 1
        except UnicodeDecodeError as e:
            logging.error(f"  idx={idx}: failed to decode JSON for sample—skipping: {e}")
        except Exception as e:
            logging.error(f"  idx={idx}: unexpected error loading sample—skipping: {e}")

    if good == 0:
        logging.warning("  No valid samples could be read from this root!")

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

    # ENV
    logging.info(
        f"ENV VARS: RANK={os.environ.get('RANK')}, "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}"
    )

    roots = [r.strip() for r in args.data_roots.split(",") if r.strip()]
    total = 0
    for root in roots:
        total += debug_root(root, args.crop_size, args.max_samples, args.annotate)
    logging.info(f"=== GRAND TOTAL across {roots}: {total} ===")


if __name__ == "__main__":
    main()
