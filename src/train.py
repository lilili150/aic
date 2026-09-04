"""Train a small SegFormer semantic-segmentation baseline.

The split CSV stores sample IDs and may contain Windows paths. This script
uses the sample ID with --data-root so the same split works on AutoDL/Linux.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import TextIO

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation

NUM_CLASSES = 9
IGNORE_INDEX = 0
IMAGE_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGE_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
CLASS_NAMES = (
    "Ignore",
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Vegetation",
    "Agricultural",
    "Vehicle",
)


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class SegmentationDataset(Dataset):
    def __init__(self, csv_path: Path, data_root: Path, image_size: int, train: bool) -> None:
        self.data_root = data_root
        self.image_size = image_size
        self.train = train
        with csv_path.open(encoding="utf-8-sig", newline="") as file:
            self.sample_ids = [row["id"] for row in csv.DictReader(file)]
        if not self.sample_ids:
            raise ValueError(f"No samples found in {csv_path}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_id = self.sample_ids[index]
        image_path = self.data_root / "images" / f"{sample_id}.png"
        mask_path = self.data_root / "masks" / f"{sample_id}.png"
        flip = self.train and random.random() < 0.5
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if flip:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            image_array = np.asarray(image, dtype=np.float32) / 255.0
            image_array = (image_array - IMAGE_MEAN) / IMAGE_STD
        with Image.open(mask_path) as mask:
            mask = mask.convert("L")
            if flip:
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            mask_array = np.asarray(mask, dtype=np.int64)
        if np.any((mask_array < 0) | (mask_array >= NUM_CLASSES)):
            raise ValueError(f"Invalid label in {mask_path}")
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array)
        return image_tensor, mask_tensor


def compute_miou(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, list[float | None]]:
    predictions = logits.argmax(dim=1)
    ious: list[float | None] = [None] * NUM_CLASSES
    for class_id in range(1, NUM_CLASSES):
        predicted = predictions == class_id
        actual = targets == class_id
        union = (predicted | actual).sum().item()
        intersection = (predicted & actual).sum().item()
        ious[class_id] = intersection / union if union else None
    valid_ious = [value for value in ious if value is not None]
    return (sum(valid_ious) / len(valid_ious) if valid_ious else 0.0), ious


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[float | None]]:
    model.eval()
    total_intersection = [0] * NUM_CLASSES
    total_union = [0] * NUM_CLASSES
    with torch.no_grad():
        for images, masks in loader:
            outputs = model(pixel_values=images.to(device)).logits
            outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            predictions = outputs.argmax(dim=1).cpu()
            for class_id in range(1, NUM_CLASSES):
                predicted = predictions == class_id
                actual = masks == class_id
                total_intersection[class_id] += int((predicted & actual).sum())
                total_union[class_id] += int((predicted | actual).sum())
    ious = [None] + [total_intersection[i] / total_union[i] if total_union[i] else None for i in range(1, NUM_CLASSES)]
    valid_ious = [value for value in ious if value is not None]
    return sum(valid_ious) / len(valid_ious), ious


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing images/ and masks/")
    parser.add_argument("--train-csv", type=Path, required=True, help="CSV containing an id column")
    parser.add_argument("--val-csv", type=Path, required=True, help="CSV containing an id column")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/exp001_baseline"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--pretrained", default="nvidia/mit-b0")
    parser.add_argument("--log-file", type=Path, help="Write stdout and stderr to this file as well as the terminal")
    args = parser.parse_args()

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_handle = None
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log_file.open("w", encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, log_handle)  # type: ignore[assignment]
        sys.stderr = Tee(sys.__stderr__, log_handle)  # type: ignore[assignment]
    started_at = time.time()
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)

    train_dataset = SegmentationDataset(args.train_csv, args.data_root, args.image_size, train=True)
    val_dataset = SegmentationDataset(args.val_csv, args.data_root, args.image_size, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    config = SegformerConfig.from_pretrained(args.pretrained, num_labels=NUM_CLASSES, semantic_loss_ignore_index=IGNORE_INDEX)
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.pretrained, config=config, ignore_mismatched_sizes=True
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    scaler_enabled = device.type == "cuda" and not args.no_amp
    run_config = {
        "model": "SegFormer MIT-B0",
        "pretrained": args.pretrained,
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
        "class_names": list(CLASS_NAMES),
        "parameters": parameter_count,
        "trainable_parameters": trainable_parameter_count,
        "device": str(device),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "amp": scaler_enabled,
    }
    write_json(args.output_dir / "run_config.json", run_config)
    print(json.dumps({"run_config": run_config}, ensure_ascii=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    best_miou = -1.0
    history: list[dict[str, object]] = []
    history_path = args.output_dir / "history.json"
    history_jsonl_path = args.output_dir / "history.jsonl"
    history_jsonl = history_jsonl_path.open("w", encoding="utf-8")

    print(f"device={device} train={len(train_dataset)} val={len(val_dataset)} image_size={args.image_size}")
    for epoch in range(1, args.epochs + 1):
        epoch_started_at = time.time()
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(pixel_values=images.to(device)).logits
                logits = torch.nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = torch.nn.functional.cross_entropy(logits, masks.to(device), ignore_index=IGNORE_INDEX)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
        miou, ious = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "val_miou": miou,
            "ious": ious,
            "elapsed_seconds": time.time() - epoch_started_at,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        history_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        history_jsonl.flush()
        write_json(history_path, history)
        if miou > best_miou:
            best_miou = miou
            model.save_pretrained(args.output_dir / "best_model")
            torch.save({"epoch": epoch, "val_miou": miou}, args.output_dir / "best_metrics.pt")
            write_json(args.output_dir / "best_metrics.json", {"epoch": epoch, "val_miou": miou, "ious": ious})
    history_jsonl.close()
    elapsed_seconds = time.time() - started_at
    best_record = max(history, key=lambda item: float(item["val_miou"]))
    summary = {
        "best_epoch": best_record["epoch"],
        "best_miou": best_miou,
        "final_miou": history[-1]["val_miou"],
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_seconds / 60,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    if log_handle:
        log_handle.flush()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())