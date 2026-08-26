"""
LoveDA dataset validation before training.

Never starts training on an invalid or empty dataset.
Does not download data. Does not invent samples.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}
EXPECTED_CLASSES = {0, 1, 2, 3, 4, 5, 6, 7}


@dataclass
class SplitReport:
    name: str
    n_images: int = 0
    n_masks: int = 0
    n_pairs: int = 0
    unpaired_images: List[str] = field(default_factory=list)
    unpaired_masks: List[str] = field(default_factory=list)
    corrupt: List[str] = field(default_factory=list)
    class_counts: Dict[int, int] = field(default_factory=dict)
    sample_shape: Optional[Tuple[int, ...]] = None
    sample_channels: Optional[int] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class DatasetValidation:
    root: str
    ready: bool
    message: str
    train: Optional[SplitReport] = None
    val: Optional[SplitReport] = None
    classes_detected: List[int] = field(default_factory=list)

    def print_report(self) -> None:
        print("DATASET VALIDATION")
        print("-" * 18)
        print(f"Root: {self.root}")
        for split in (self.train, self.val):
            if split is None:
                continue
            print(f"{split.name.capitalize()} images: {split.n_images}")
            print(f"{split.name.capitalize()} masks:  {split.n_masks}")
            print(f"{split.name.capitalize()} pairs:  {split.n_pairs}")
            if split.sample_shape:
                print(f"{split.name.capitalize()} sample shape: {split.sample_shape}  channels={split.sample_channels}")
            if split.unpaired_images:
                print(f"  unpaired images: {len(split.unpaired_images)}")
            if split.unpaired_masks:
                print(f"  unpaired masks: {len(split.unpaired_masks)}")
            if split.corrupt:
                print(f"  corrupt: {split.corrupt[:5]}")
            if split.errors:
                for e in split.errors[:5]:
                    print(f"  error: {e}")
        if self.classes_detected:
            print("Classes detected:", self.classes_detected)
        print(f"Dataset status: {'READY' if self.ready else 'NOT READY'}")
        print(self.message)


def _list_files(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        return {}
    out = {}
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out[p.stem] = p
    return out


def _load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    from PIL import Image

    return np.array(Image.open(path))


def _validate_split(root: Path, split: str, max_probe: int = 32) -> SplitReport:
    report = SplitReport(name=split)
    img_dir = root / split / "images"
    mask_dir = root / split / "masks"
    if not img_dir.is_dir():
        report.errors.append(f"Missing directory: {img_dir}")
        return report
    if not mask_dir.is_dir():
        report.errors.append(f"Missing directory: {mask_dir}")
        return report

    images = _list_files(img_dir)
    masks = _list_files(mask_dir)
    report.n_images = len(images)
    report.n_masks = len(masks)
    common = sorted(set(images) & set(masks))
    report.n_pairs = len(common)
    report.unpaired_images = sorted(set(images) - set(masks))[:20]
    report.unpaired_masks = sorted(set(masks) - set(images))[:20]

    if report.n_pairs == 0:
        report.errors.append(f"No matching image/mask pairs in {split}")
        return report

    class_counter: Counter = Counter()
    probed = 0
    for stem in common:
        if probed >= max_probe:
            break
        try:
            img = _load_array(images[stem])
            msk = _load_array(masks[stem])
        except Exception as e:
            report.corrupt.append(f"{stem}: {e}")
            continue
        if img.ndim == 2:
            ch = 1
        elif img.ndim == 3:
            ch = img.shape[-1] if img.shape[-1] in (1, 3, 4) else img.shape[0]
        else:
            report.corrupt.append(f"{stem}: unexpected image ndim {img.ndim}")
            continue
        if report.sample_shape is None:
            report.sample_shape = img.shape
            report.sample_channels = int(ch)
        if ch not in (1, 3, 4):
            report.errors.append(f"{stem}: unsupported channels={ch}")
        if msk.ndim == 3:
            msk = msk[..., 0] if msk.shape[-1] <= 4 else msk[0]
        if msk.size == 0:
            report.corrupt.append(f"{stem}: empty mask")
            continue
        vals = np.unique(msk.astype(np.int64))
        for v in vals.tolist():
            class_counter[int(v)] += 1
        probed += 1

    report.class_counts = dict(class_counter)
    return report


def validate_loveda(root: str | Path = "data/datasets/LoveDA") -> DatasetValidation:
    root = Path(root)
    if not root.is_dir():
        return DatasetValidation(
            root=str(root),
            ready=False,
            message=f"Dataset root does not exist: {root}. "
            f"Download LoveDA (CC BY-NC-SA 4.0) and place train/val images+masks here.",
        )

    train = _validate_split(root, "train")
    val = _validate_split(root, "val")
    classes: Set[int] = set()
    classes.update(train.class_counts.keys())
    classes.update(val.class_counts.keys())

    ready = (
        train.n_pairs > 0
        and val.n_pairs > 0
        and not train.errors
        and not val.errors
        and len(train.corrupt) < max(1, train.n_pairs // 2)
    )
    if ready:
        msg = (
            f"READY — train pairs={train.n_pairs}, val pairs={val.n_pairs}. "
            "License: LoveDA CC BY-NC-SA 4.0 (non-commercial research)."
        )
    else:
        problems = train.errors + val.errors
        if train.n_pairs == 0:
            problems.append("train has 0 pairs")
        if val.n_pairs == 0:
            problems.append("val has 0 pairs")
        msg = "NOT READY — " + ("; ".join(problems) if problems else "unknown issue")

    return DatasetValidation(
        root=str(root),
        ready=ready,
        message=msg,
        train=train,
        val=val,
        classes_detected=sorted(classes),
    )
