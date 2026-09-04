"""Run a cheap environment and segmentation-data smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_sample(images_dir: Path, masks_dir: Path, image_name: str | None, size: int) -> tuple[torch.Tensor, torch.Tensor, str]:
    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {images_dir}")

    image_path = images_dir / image_name if image_name else image_paths[0]
    mask_path = masks_dir / image_path.name
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Matching mask does not exist: {mask_path}")

    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
    with Image.open(mask_path) as mask:
        mask = mask.convert("L").resize((size, size), Image.Resampling.NEAREST)
        mask_array = np.asarray(mask, dtype=np.int64)

    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
    return image_tensor, mask_tensor, image_path.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--image-name", help="Optional PNG name; defaults to the first sorted image")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()

    if not args.images.is_dir() or not args.masks.is_dir():
        raise NotADirectoryError("Both --images and --masks must be existing directories")
    if args.size <= 0:
        raise ValueError("--size must be positive")

    print("torch:", torch.__version__)
    print("torch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    image_tensor, mask_tensor, image_name = load_sample(args.images, args.masks, args.image_name, args.size)
    unique_values = torch.unique(mask_tensor).tolist()
    invalid_values = [value for value in unique_values if value < 0 or value > 8]
    if invalid_values:
        raise ValueError(f"Invalid mask values in {image_name}: {invalid_values}")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    image_tensor = image_tensor.to(device)
    mask_tensor = mask_tensor.to(device)

    model = torch.nn.Conv2d(3, 9, kernel_size=1).to(device)
    logits = model(image_tensor)
    loss = torch.nn.functional.cross_entropy(logits, mask_tensor)

    print("sample:", image_name)
    print("image tensor:", tuple(image_tensor.shape), image_tensor.dtype, image_tensor.device)
    print("mask tensor:", tuple(mask_tensor.shape), mask_tensor.dtype, mask_tensor.device)
    print("mask values:", unique_values)
    print("logits:", tuple(logits.shape))
    print("loss:", float(loss.detach().cpu()))
    print("SMOKE TEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())