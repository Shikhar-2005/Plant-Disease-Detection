"""
setup_dataset.py — Phase 2: Download, verify, and split the PlantVillage dataset.

Steps this script performs:
  1. Check for Kaggle API credentials (kaggle.json)
  2. Download the dataset via Kaggle API
  3. Unzip into data/plantvillage/
  4. Verify 38 class folders exist in both train/ and valid/
  5. Carve a test/ split (50% of valid/ moved to test/)
  6. Print final class counts

Usage:
    python setup_dataset.py

Requirements:
    pip install kaggle     (already in requirements.txt via huggingface-hub dep chain)
    Kaggle API token at:  C:/Users/<you>/.kaggle/kaggle.json
                       OR  set env vars KAGGLE_USERNAME and KAGGLE_KEY
"""

import os
import shutil
import zipfile
import random
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
KAGGLE_DATASET = "vipoooool/new-plant-diseases-dataset"
DATA_DIR       = Path("data/plantvillage")
ZIP_PATH       = Path("data/plant-diseases-raw.zip")
EXPECTED_CLASSES = 38
TEST_SPLIT_RATIO  = 0.5   # 50% of valid/ → test/
RANDOM_SEED       = 42

# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(msg: str):
    print(f"\n{'-'*60}")
    print(f"  {msg}")
    print(f"{'-'*60}")


def check_kaggle_credentials():
    """Verify Kaggle API credentials are in place."""
    kaggle_json   = Path.home() / ".kaggle" / "kaggle.json"
    access_token  = Path.home() / ".kaggle" / "access_token"
    env_ok = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    env_token = os.environ.get("KAGGLE_API_TOKEN")

    if kaggle_json.exists() or access_token.exists() or env_ok or env_token:
        print("[OK] Kaggle credentials found.")
        return True

    print("[!!] Kaggle credentials NOT found.")
    print()
    print("To fix, do ONE of the following:")
    print()
    print("  Option A - access_token (your current setup):")
    print("    Save token to: C:\\Users\\shikh\\.kaggle\\access_token")
    print()
    print("  Option B - kaggle.json:")
    print("    1. Go to https://www.kaggle.com/settings - API - Create New Token")
    print("    2. Save to: C:\\Users\\shikh\\.kaggle\\kaggle.json")
    print()
    print("Then re-run this script.")
    return False


def download_dataset():
    """Download the PlantVillage dataset from Kaggle."""
    banner("Step 1/5 — Downloading dataset from Kaggle")

    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        print(f"[SKIP] {DATA_DIR} already exists and is non-empty -- skipping download.")
        return

    Path("data").mkdir(exist_ok=True)

    print(f"Downloading: {KAGGLE_DATASET}")
    print("This is ~3.5 GB -- may take several minutes on a slow connection...\n")

    # kaggle datasets download -d <dataset> -p data/ --unzip
    ret = os.system(
        f".venv\\Scripts\\python -m kaggle datasets download "
        f"-d {KAGGLE_DATASET} -p data/ --unzip"
    )
    if ret != 0:
        raise RuntimeError("Kaggle download failed. Check credentials and internet connection.")
    print("[OK] Download complete.")


def find_plantvillage_root() -> Path:
    """
    The Kaggle dataset unzips into a nested folder.
    Locate the folder that actually contains train/ and valid/.
    """
    # Common patterns after unzip
    candidates = [
        DATA_DIR,
        Path("data") / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)",
        Path("data") / "plant-disease-recognition-dataset",
        Path("data") / "new plant diseases dataset(augmented)",
    ]
    for path in candidates:
        if (path / "train").exists() and (path / "valid").exists():
            return path

    # Fallback: walk data/ to find first directory containing train/ and valid/
    for root, dirs, _ in os.walk("data"):
        r = Path(root)
        if (r / "train").exists() and (r / "valid").exists():
            return r

    raise FileNotFoundError(
        "Could not locate train/ and valid/ folders inside data/. "
        "Check dataset structure manually."
    )


def reorganise_to_data_dir(source: Path):
    """Move train/ and valid/ into DATA_DIR if they are in a nested location."""
    banner("Step 2/5 — Reorganising dataset structure")

    if source == DATA_DIR:
        print(f"[OK] Dataset already at {DATA_DIR} -- no move needed.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        src = source / split
        dst = DATA_DIR / split
        if src.exists() and not dst.exists():
            print(f"  Moving {src} -> {dst}")
            shutil.move(str(src), str(dst))

    print("[OK] Reorganisation complete.")


def verify_classes(expected: int = EXPECTED_CLASSES) -> list[str]:
    """Verify that both train/ and valid/ contain the expected number of classes."""
    banner("Step 3/5 — Verifying class folders")

    for split in ("train", "valid"):
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing directory: {split_dir}")

        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        count   = len(classes)
        status  = "[OK]" if count == expected else "[!!]"
        print(f"  {status} {split}/  ->  {count} classes")

        if count != expected:
            print(f"     Expected {expected}, got {count}.")
            print(f"     Classes found: {classes}")

    # Return train class names (source of truth)
    return sorted([d.name for d in (DATA_DIR / "train").iterdir() if d.is_dir()])


def create_test_split(class_names: list[str], ratio: float = TEST_SPLIT_RATIO):
    """
    Carve test/ from valid/ by moving ratio% of each class's images.
    Preserves class balance.
    """
    banner("Step 4/5 — Creating test/ split")

    test_dir = DATA_DIR / "test"
    if test_dir.exists() and any(test_dir.iterdir()):
        print(f"[SKIP] test/ already exists and is non-empty -- skipping split.")
        return

    random.seed(RANDOM_SEED)

    for cls in class_names:
        valid_cls = DATA_DIR / "valid" / cls
        test_cls  = test_dir  / cls
        test_cls.mkdir(parents=True, exist_ok=True)

        images = sorted([
            f for f in valid_cls.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        n_test = int(len(images) * ratio)
        to_move = random.sample(images, n_test)

        for img in to_move:
            shutil.move(str(img), str(test_cls / img.name))

    print(f"[OK] Test split created (moved {ratio*100:.0f}% of each valid/ class).")


def print_stats(class_names: list[str]):
    """Print final image counts per split."""
    banner("Step 5/5 — Final dataset statistics")

    total = 0
    print(f"  {'Class':<45} {'Train':>7} {'Valid':>7} {'Test':>7}")
    print(f"  {'-'*45} {'-'*7} {'-'*7} {'-'*7}")

    for cls in class_names:
        counts = {}
        for split in ("train", "valid", "test"):
            d = DATA_DIR / split / cls
            if d.exists():
                counts[split] = len(list(d.iterdir()))
            else:
                counts[split] = 0

        row_total = sum(counts.values())
        total += row_total
        print(f"  {cls:<45} {counts['train']:>7} {counts['valid']:>7} {counts['test']:>7}")

    print(f"\n  {'TOTAL':<45} {'':>7} {'':>7} {'':>7}")
    print(f"  Total images across all splits: {total:,}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\nPlantGuard -- Dataset Setup Script")
    print("====================================\n")

    if not check_kaggle_credentials():
        return

    download_dataset()

    source = find_plantvillage_root()
    reorganise_to_data_dir(source)

    class_names = verify_classes(expected=EXPECTED_CLASSES)
    create_test_split(class_names, ratio=TEST_SPLIT_RATIO)
    print_stats(class_names)

    print("\n[OK] Dataset setup complete! You can now run the EDA notebook (Phase 3).")
    print(f"   Dataset location: {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
