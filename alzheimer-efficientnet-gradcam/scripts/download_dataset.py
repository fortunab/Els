from __future__ import annotations

import os
import shutil
from pathlib import Path

import kagglehub

DATASET = "lukechugh/best-alzheimer-mri-dataset-99-accuracy"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                continue
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main():
    path = Path(kagglehub.dataset_download(DATASET))
    print(f"Downloaded dataset to KaggleHub cache: {path}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    copy_tree_contents(path, DATA_DIR)
    print(f"Copied dataset contents to: {DATA_DIR}")
    print("Expected final layout: data/train/<class folders> and data/test/<class folders>.")
    print("If the downloaded archive has an extra nested folder, move train/ and test/ directly under data/.")


if __name__ == "__main__":
    main()
