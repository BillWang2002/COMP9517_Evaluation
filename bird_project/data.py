"""Strict YOLO bird annotations and the A0-compatible detection dataset.

Annotation inspection does not import PyTorch, so it can run before model setup.
Empty, present label files produce empty targets; missing labels are errors.
An empty annotation is not proof that an image is visually free of birds.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
SPLITS = frozenset({"train", "val", "test"})
EDGE_TOLERANCE = 1e-6  # Six-decimal YOLO serialization can round an edge outward.


class AnnotationError(ValueError):
    """An image/annotation pair is incomplete or has structurally invalid labels."""


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path


def collect_samples(root: str | Path, split: str) -> list[Sample]:
    """Return sorted, one-to-one image/label pairs, rejecting pairing errors."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; use train, val or test.")
    root = Path(root)
    image_dir, label_dir = root / "images" / split, root / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(
            f"Expected both {image_dir} and {label_dir}. "
            "Pass the dataset directory containing images/ and labels/."
        )
    paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise AnnotationError(f"No JPEG/PNG images found in {image_dir}.")
    image_stems: dict[str, Path] = {}
    issues = []
    for path in paths:
        if path.stem in image_stems:
            issues.append(f"Duplicate image stem: {image_stems[path.stem]} and {path}")
        image_stems[path.stem] = path
        label = label_dir / f"{path.stem}.txt"
        if not label.is_file():
            issues.append(
                f"Missing label: {label} for {path.name}. "
                "An empty-target image must have a present, empty .txt file."
            )
    for label in sorted(label_dir.iterdir()):
        if label.is_file() and label.suffix.lower() == ".txt" and label.stem not in image_stems:
            issues.append(f"Orphan label without a matching image: {label}")
    if issues:
        shown = "\n".join(issues[:30])
        extra = f"\n... and {len(issues) - 30} more pairing errors." if len(issues) > 30 else ""
        raise AnnotationError(shown + extra)
    return [Sample(path, label_dir / f"{path.stem}.txt") for path in paths]


def parse_yolo_file(path: str | Path, width: int, height: int) -> list[list[float]]:
    """Read class-0 normalized YOLO boxes as original-image pixel XYXY.

    Values must be finite; centers must be in [0, 1], sizes in (0, 1], and
    box edges within the image. Only edge roundoff <= 1e-6 is clipped, as in
    A0. Invalid rows are rejected rather than silently dropped or repaired.
    """
    path = Path(path)
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width} x {height}.")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing label {path}; empty labels must be actual empty files."
        )
    boxes: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        prefix = f"{path}:{line_number}"
        fields = line.split()
        if len(fields) != 5:
            raise AnnotationError(f"{prefix}: expected 5 YOLO fields, got {len(fields)}.")
        try:
            cls, xc, yc, bw, bh = map(float, fields)
        except ValueError as exc:
            raise AnnotationError(f"{prefix}: YOLO fields must be numeric: {line!r}.") from exc
        if not all(math.isfinite(value) for value in (cls, xc, yc, bw, bh)):
            raise AnnotationError(f"{prefix}: all YOLO fields must be finite.")
        if cls != 0:
            raise AnnotationError(f"{prefix}: expected bird class 0, got {cls}.")
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise AnnotationError(
                f"{prefix}: normalized centers must be in [0,1] and widths/heights in (0,1]."
            )
        left, top, right, bottom = xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2
        if min(left, top) < -EDGE_TOLERANCE or max(right, bottom) > 1 + EDGE_TOLERANCE:
            raise AnnotationError(
                f"{prefix}: box edges extend outside the image: "
                f"({left:.8g}, {top:.8g}, {right:.8g}, {bottom:.8g})."
            )
        box = [
            max(0.0, left * width), max(0.0, top * height),
            min(float(width), right * width), min(float(height), bottom * height),
        ]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise AnnotationError(f"{prefix}: box has zero area at image precision.")
        boxes.append(box)
    return boxes


class BirdDataset:
    """Map-style dataset accepted by torch DataLoader; tensors match valid A0 data.

    PyTorch is imported on first item access, keeping the auditing helpers usable
    without the training dependencies. No augmentation or resize is applied here.
    """

    def __init__(self, root: str | Path, split: str):
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        self.samples = collect_samples(self.root, split)
        self.paths = [sample.image_path for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import numpy as np
        import torch

        sample = self.samples[index]
        with Image.open(sample.image_path) as original:
            image = original.convert("RGB")
        width, height = image.size
        boxes = parse_yolo_file(sample.label_path, width, height)
        image_tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).contiguous().float() / 255.0
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor(index, dtype=torch.int64),
        }
        return image_tensor, target


def collate(batch):
    """Detection batches keep variable-length targets as individual dictionaries."""
    return tuple(zip(*batch))
