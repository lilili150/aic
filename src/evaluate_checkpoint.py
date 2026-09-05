"""Evaluate a saved SegFormer checkpoint on the fixed validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from train import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, SegmentationDataset


def evaluate_checkpoint(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[float | None], float]:
    model.eval()
    total_intersection = [0] * NUM_CLASSES
    total_union = [0] * NUM_CLASSES
    total_valid_pixels = 0
    total_predicted_ignore = 0
    with torch.no_grad():
        for images, masks in loader:
            outputs = model(pixel_values=images.to(device)).logits
            outputs = torch.nn.functional.interpolate(
                outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
            predictions = outputs.argmax(dim=1).cpu()
            valid = masks != IGNORE_INDEX
            total_valid_pixels += int(valid.sum())
            total_predicted_ignore += int(((predictions == IGNORE_INDEX) & valid).sum())
            for class_id in range(1, NUM_CLASSES):
                predicted = (predictions == class_id) & valid
                actual = (masks == class_id) & valid
                total_intersection[class_id] += int((predicted & actual).sum())
                total_union[class_id] += int((predicted | actual).sum())
    ious = [None] + [
        total_intersection[class_id] / total_union[class_id] if total_union[class_id] else None
        for class_id in range(1, NUM_CLASSES)
    ]
    valid_ious = [value for value in ious if value is not None]
    miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    predicted_ignore_ratio = total_predicted_ignore / total_valid_pixels if total_valid_pixels else 0.0
    return miou, ious, predicted_ignore_ratio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    dataset = SegmentationDataset(args.val_csv, args.data_root, args.image_size, train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = SegformerForSemanticSegmentation.from_pretrained(args.checkpoint).to(device)
    miou, ious, predicted_ignore_ratio = evaluate_checkpoint(model, loader, device)
    result = {
        "checkpoint": str(args.checkpoint),
        "image_size": args.image_size,
        "validation_samples": len(dataset),
        "miou": miou,
        "ious": ious,
        "class_names": list(CLASS_NAMES),
        "predicted_ignore_ratio_on_valid_pixels": predicted_ignore_ratio,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
