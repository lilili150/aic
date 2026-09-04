"""Create a fixed train/validation split and summarize mask class distribution."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


def load_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[str, Path, Path]]:
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = {path.stem: path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in allowed}
    masks = {path.stem: path for path in masks_dir.iterdir() if path.is_file() and path.suffix.lower() in allowed}
    return [(name, images[name], masks[name]) for name in sorted(images.keys() & masks.keys())]


def summarize(pairs: list[tuple[str, Path, Path]], label_names: dict[int, str]) -> dict[str, Any]:
    image_counts: dict[int, int] = defaultdict(int)
    pixel_counts: dict[int, int] = defaultdict(int)
    errors: list[dict[str, str]] = []
    for name, image_path, mask_path in pairs:
        try:
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.size != mask.size:
                    errors.append({"name": name, "error": f"size mismatch: {image.size} vs {mask.size}"})
                values, counts = np.unique(np.asarray(mask), return_counts=True)
                for value, count in zip(values, counts):
                    class_id = int(value)
                    image_counts[class_id] += 1
                    pixel_counts[class_id] += int(count)
        except (OSError, UnidentifiedImageError) as error:
            errors.append({"name": name, "error": str(error)})

    return {
        "image_count": len(pairs),
        "class_image_counts": {
            str(class_id): image_counts.get(class_id, 0)
            for class_id in sorted(set(label_names) | set(image_counts))
        },
        "class_pixel_counts": {
            str(class_id): pixel_counts.get(class_id, 0)
            for class_id in sorted(set(label_names) | set(pixel_counts))
        },
        "errors": errors,
    }


def parse_labels(label_file: Path | None) -> dict[int, str]:
    if label_file is None or not label_file.exists():
        return {}
    labels: dict[int, str] = {}
    for line in label_file.read_text(encoding="utf-8").splitlines():
        line = line.strip().rstrip(",")
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip().isdigit():
            labels[int(key.strip())] = value.strip().strip("\"'")
    return labels


def write_split(path: Path, pairs: list[tuple[str, Path, Path]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "image", "mask"])
        for name, image_path, mask_path in pairs:
            writer.writerow([name, str(image_path), str(mask_path)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1:
        parser.error("--val-ratio must be between 0 and 1")

    pairs = load_pairs(args.images, args.masks)
    if len(pairs) < 2:
        parser.error("at least two paired samples are required")
    random.Random(args.seed).shuffle(pairs)
    val_count = max(1, round(len(pairs) * args.val_ratio))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]
    label_names = parse_labels(args.labels)

    write_split(args.output_dir / "train.csv", train_pairs)
    write_split(args.output_dir / "val.csv", val_pairs)
    report = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "train": summarize(train_pairs, label_names),
        "validation": summarize(val_pairs, label_names),
    }
    (args.output_dir / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Seed: {args.seed}")
    print(f"Train samples: {len(train_pairs)}")
    print(f"Validation samples: {len(val_pairs)}")
    print(f"Train class image counts: {report['train']['class_image_counts']}")
    print(f"Validation class image counts: {report['validation']['class_image_counts']}")
    print(f"Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())