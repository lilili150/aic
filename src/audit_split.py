"""Audit a fixed segmentation split without changing data or training a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


def read_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return [row["id"] for row in csv.DictReader(file)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(ids: list[str], data_root: Path, sample_limit: int) -> dict[str, Any]:
    class_pixels: Counter[int] = Counter()
    class_images: Counter[int] = Counter()
    errors: list[dict[str, str]] = []
    image_hashes: dict[str, str] = {}
    mask_hashes: dict[str, str] = {}
    for index, sample_id in enumerate(ids):
        image_path = data_root / "images" / f"{sample_id}.png"
        mask_path = data_root / "masks" / f"{sample_id}.png"
        if not image_path.exists() or not mask_path.exists():
            errors.append({"id": sample_id, "error": "missing image or mask"})
            continue
        try:
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.size != mask.size:
                    errors.append({"id": sample_id, "error": f"size mismatch: {image.size} vs {mask.size}"})
                values, counts = np.unique(np.asarray(mask.convert("L")), return_counts=True)
                for value, count in zip(values, counts):
                    class_id = int(value)
                    class_pixels[class_id] += int(count)
                    class_images[class_id] += 1
            if index < sample_limit:
                image_hashes[sample_id] = sha256(image_path)
                mask_hashes[sample_id] = sha256(mask_path)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            errors.append({"id": sample_id, "error": str(error)})
    total_pixels = sum(class_pixels.values())
    return {
        "sample_count": len(ids),
        "class_pixel_counts": {str(key): value for key, value in sorted(class_pixels.items())},
        "class_pixel_ratios": {
            str(key): value / total_pixels if total_pixels else 0.0
            for key, value in sorted(class_pixels.items())
        },
        "class_image_counts": {str(key): value for key, value in sorted(class_images.items())},
        "errors": errors,
        "image_hashes_sample": image_hashes,
        "mask_hashes_sample": mask_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-limit", type=int, default=0, help="Hash all samples when 0; otherwise hash only the first N per split")
    args = parser.parse_args()
    train_ids = read_ids(args.train_csv)
    val_ids = read_ids(args.val_csv)
    train_set = set(train_ids)
    val_set = set(val_ids)
    duplicate_ids = sorted(train_set & val_set)
    duplicate_train_ids = sorted(name for name, count in Counter(train_ids).items() if count > 1)
    duplicate_val_ids = sorted(name for name, count in Counter(val_ids).items() if count > 1)
    hash_limit = args.hash_limit if args.hash_limit > 0 else max(len(train_ids), len(val_ids))
    train_summary = summarize(train_ids, args.data_root, hash_limit)
    val_summary = summarize(val_ids, args.data_root, hash_limit)
    train_hashes = train_summary["image_hashes_sample"]
    val_hashes = val_summary["image_hashes_sample"]
    train_mask_hashes = train_summary["mask_hashes_sample"]
    val_mask_hashes = val_summary["mask_hashes_sample"]
    duplicate_image_hashes = sorted({train_hashes[sample_id] for sample_id in train_hashes} & set(val_hashes.values()))
    duplicate_mask_hashes = sorted({train_mask_hashes[sample_id] for sample_id in train_mask_hashes} & set(val_mask_hashes.values()))
    report = {
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv),
        "data_root": str(args.data_root),
        "train": train_summary,
        "validation": val_summary,
        "id_overlap": duplicate_ids,
        "duplicate_ids_within_train": duplicate_train_ids,
        "duplicate_ids_within_validation": duplicate_val_ids,
        "cross_split_duplicate_image_hashes": duplicate_image_hashes,
        "cross_split_duplicate_mask_hashes": duplicate_mask_hashes,
        "hash_scope": "all samples" if args.hash_limit <= 0 else f"first {args.hash_limit} samples per split",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "train_samples": len(train_ids),
        "validation_samples": len(val_ids),
        "id_overlap_count": len(duplicate_ids),
        "duplicate_train_id_count": len(duplicate_train_ids),
        "duplicate_validation_id_count": len(duplicate_val_ids),
        "cross_split_duplicate_image_hash_count": len(duplicate_image_hashes),
        "cross_split_duplicate_mask_hash_count": len(duplicate_mask_hashes),
        "train_errors": len(train_summary["errors"]),
        "validation_errors": len(val_summary["errors"]),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
