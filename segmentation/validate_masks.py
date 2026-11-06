"""Validate manually created binary masks against source images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def image_files(root: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def validate(images_root: Path, masks_root: Path) -> int:
    images = image_files(images_root)
    masks = image_files(masks_root)
    missing = sorted(set(images) - set(masks))
    orphaned = sorted(set(masks) - set(images))
    errors: list[str] = []
    for stem in sorted(set(images) & set(masks)):
        with Image.open(images[stem]) as image, Image.open(masks[stem]) as mask:
            if image.size != mask.size:
                errors.append(f"{stem}: image size {image.size} != mask size {mask.size}")
            mask_array = np.asarray(mask.convert("L"))
            values = set(np.unique(mask_array).tolist())
            if not values <= {0, 255}:
                errors.append(f"{stem}: mask values must be only 0 and 255, got {sorted(values)}")
    if missing:
        errors.append(f"Missing masks ({len(missing)}): {', '.join(missing[:5])}")
    if orphaned:
        errors.append(f"Orphan masks ({len(orphaned)}): {', '.join(orphaned[:5])}")
    if errors:
        raise ValueError("\n".join(errors))
    print(f"Validated {len(images)} image/mask pairs.")
    return len(images)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    args = parser.parse_args()
    validate(args.images, args.masks)


if __name__ == "__main__":
    main()
