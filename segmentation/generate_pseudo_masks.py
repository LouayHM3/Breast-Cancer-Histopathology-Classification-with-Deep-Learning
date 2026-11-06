"""Generate heuristic pseudo-masks for a technical segmentation demo.

These masks are not expert annotations and must not be used to claim medical
segmentation performance. The heuristic selects strongly colored/dark tissue
regions and applies simple morphology-free cleanup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


EXTENSIONS = {".png", ".jpg", ".jpeg"}


def pseudo_mask(image: Image.Image, percentile: float = 65.0) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    brightness = rgb.mean(axis=2)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    score = (1.0 - brightness) * 0.6 + saturation * 0.4
    threshold = np.percentile(score, percentile)
    mask = (score >= threshold).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")


def generate(images_root: Path, masks_root: Path, limit: int | None) -> int:
    paths = sorted(
        path
        for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if limit is not None:
        paths = paths[:limit]
    masks_root.mkdir(parents=True, exist_ok=True)
    for path in paths:
        with Image.open(path) as image:
            mask = pseudo_mask(image)
        mask.save(masks_root / f"{path.stem}.png")
    return len(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    count = generate(args.images, args.masks, args.limit)
    print(f"Generated {count} pseudo-masks.")
    print("WARNING: These are heuristic demo artifacts, not medical annotations.")


if __name__ == "__main__":
    main()
