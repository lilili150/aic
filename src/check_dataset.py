"""Validate an image/mask semantic-segmentation dataset without modifying it."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


def parse_labels(label_file: Path | None) -> dict[int, str]:
    """Read the project's LABEL_NAMES mapping when it is available."""
    if label_file is None or not label_file.exists():
        return {}

    labels: dict[int, str] = {}
    for line in label_file.read_text(encoding="utf-8").splitlines():
        line = line.strip().rstrip(",")
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key.isdigit():
            labels[int(key)] = value
    return labels


def image_files(directory: Path) -> dict[str, Path]:
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return {path.stem: path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in allowed}


def inspect_dataset(images_dir: Path, masks_dir: Path, label_file: Path | None, sample_count: int) -> dict[str, Any]:
    images = image_files(images_dir)
    masks = image_files(masks_dir)
    label_names = parse_labels(label_file)
    image_names = set(images)
    mask_names = set(masks)
    common_names = sorted(image_names & mask_names)

    report: dict[str, Any] = {
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "image_count": len(images),
        "mask_count": len(masks),
        "missing_masks": sorted(image_names - mask_names),
        "missing_images": sorted(mask_names - image_names),
        "image_modes": defaultdict(int),
        "mask_modes": defaultdict(int),
        "image_sizes": defaultdict(int),
        "mask_sizes": defaultdict(int),
        "corrupt_images": [],
        "corrupt_masks": [],
        "size_mismatches": [],
        "mask_values": {},
        "invalid_mask_values": {},
        "class_image_counts": defaultdict(int),
        "class_pixel_counts": defaultdict(int),
        "samples": common_names[:sample_count],
    }
    allowed_values = set(label_names) if label_names else set(range(9))

    for name, path in sorted(images.items()):
        try:
            with Image.open(path) as image:
                report["image_modes"][image.mode] += 1
                report["image_sizes"][f"{image.width}x{image.height}"] += 1
        except (OSError, UnidentifiedImageError):
            report["corrupt_images"].append(str(path))

    for name, path in sorted(masks.items()):
        try:
            with Image.open(path) as mask:
                report["mask_modes"][mask.mode] += 1
                report["mask_sizes"][f"{mask.width}x{mask.height}"] += 1
                values, counts = np.unique(np.asarray(mask), return_counts=True)
                value_counts = {str(int(value)): int(count) for value, count in zip(values, counts)}
                report["mask_values"][name] = value_counts
                invalid = {key: value for key, value in value_counts.items() if int(key) not in allowed_values}
                if invalid:
                    report["invalid_mask_values"][name] = invalid
                for key, count in value_counts.items():
                    report["class_pixel_counts"][key] += count
                for key in value_counts:
                    report["class_image_counts"][key] += 1
        except (OSError, UnidentifiedImageError):
            report["corrupt_masks"].append(str(path))

    for name in common_names:
        try:
            with Image.open(images[name]) as image, Image.open(masks[name]) as mask:
                if image.size != mask.size:
                    report["size_mismatches"].append(
                        {"name": name, "image_size": list(image.size), "mask_size": list(mask.size)}
                    )
        except (OSError, UnidentifiedImageError):
            pass

    report["image_modes"] = dict(report["image_modes"])
    report["mask_modes"] = dict(report["mask_modes"])
    report["image_sizes"] = dict(report["image_sizes"])
    report["mask_sizes"] = dict(report["mask_sizes"])
    report["class_image_counts"] = dict(sorted(report["class_image_counts"].items(), key=lambda item: int(item[0])))
    report["class_pixel_counts"] = dict(sorted(report["class_pixel_counts"].items(), key=lambda item: int(item[0])))
    report["class_names"] = {str(key): value for key, value in sorted(label_names.items())}
    report["status"] = "PASS" if not any(
        (
            report["missing_masks"],
            report["missing_images"],
            report["corrupt_images"],
            report["corrupt_masks"],
            report["size_mismatches"],
            report["invalid_mask_values"],
        )
    ) else "FAIL"
    return report


def print_summary(report: dict[str, Any]) -> None:
    print(f"Status: {report['status']}")
    print(f"Images: {report['image_count']}")
    print(f"Masks: {report['mask_count']}")
    print(f"Missing masks: {len(report['missing_masks'])}")
    print(f"Missing images: {len(report['missing_images'])}")
    print(f"Corrupt images: {len(report['corrupt_images'])}")
    print(f"Corrupt masks: {len(report['corrupt_masks'])}")
    print(f"Size mismatches: {len(report['size_mismatches'])}")
    print(f"Invalid mask files: {len(report['invalid_mask_values'])}")
    print(f"Image modes: {report['image_modes']}")
    print(f"Mask modes: {report['mask_modes']}")
    print(f"Image sizes: {report['image_sizes']}")
    print(f"Mask sizes: {report['mask_sizes']}")
    print(f"Class image counts: {report['class_image_counts']}")
    print(f"Class pixel counts: {report['class_pixel_counts']}")
    print(f"Sample files: {report['samples']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Directory containing input images")
    parser.add_argument("--masks", type=Path, required=True, help="Directory containing label masks")
    parser.add_argument("--labels", type=Path, help="Optional Label.txt file")
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    parser.add_argument("--sample-count", type=int, default=20, help="Number of paired names to list as samples")
    args = parser.parse_args()

    for directory in (args.images, args.masks):
        if not directory.is_dir():
            parser.error(f"Directory does not exist: {directory}")
    if args.sample_count < 0:
        parser.error("--sample-count must be non-negative")

    report = inspect_dataset(args.images, args.masks, args.labels, args.sample_count)
    print_summary(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report written to: {args.report}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())