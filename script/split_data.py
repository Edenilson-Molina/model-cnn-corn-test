"""
Etapa 2 del pipeline de datos: data/raw/ → data/train/ + data/val/ + data/test/

Lee data/raw/{clase}/, balancea a LIMIT_PER_CLASS imágenes con semilla fija,
y distribuye en train (70%) / val (15%) / test (15%).
El excedente se mueve a data/unused/.
"""

import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import BASE_DIR, CLASSES, SEED

DATA_DIR   = Path(BASE_DIR) / "data"
RAW_DIR    = DATA_DIR / "raw"
TRAIN_DIR  = DATA_DIR / "train"
VAL_DIR    = DATA_DIR / "val"
TEST_DIR   = DATA_DIR / "test"
UNUSED_DIR = DATA_DIR / "unused"

LIMIT_PER_CLASS = 600
TRAIN_COUNT = 420   # 70%
VAL_COUNT   = 90    # 15%
TEST_COUNT  = 90    # 15%

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def move_file(src: Path, dst_dir: Path) -> None:
    dst = dst_dir / src.name
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))


def split_class(cls: str) -> dict:
    raw_cls = RAW_DIR / cls

    if not raw_cls.exists():
        print(f"  [ERROR] data/raw/{cls}/ no encontrada. Ejecuta collect_data.py primero.")
        return {}

    all_files = sorted([
        f for f in raw_cls.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ])

    total = len(all_files)
    if total < LIMIT_PER_CLASS:
        print(
            f"  [ADVERTENCIA] {cls}: solo {total} imágenes disponibles "
            f"(se necesitan {LIMIT_PER_CLASS}). Se usarán todas."
        )

    random.shuffle(all_files)

    selected = all_files[:LIMIT_PER_CLASS]
    unused   = all_files[LIMIT_PER_CLASS:]

    train_files = selected[:TRAIN_COUNT]
    val_files   = selected[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]
    test_files  = selected[TRAIN_COUNT + VAL_COUNT:]

    for dst in [TRAIN_DIR / cls, VAL_DIR / cls, TEST_DIR / cls, UNUSED_DIR / cls]:
        dst.mkdir(parents=True, exist_ok=True)

    for f in train_files:
        move_file(f, TRAIN_DIR / cls)
    for f in val_files:
        move_file(f, VAL_DIR / cls)
    for f in test_files:
        move_file(f, TEST_DIR / cls)
    for f in unused:
        move_file(f, UNUSED_DIR / cls)

    return {
        "train":  len(train_files),
        "val":    len(val_files),
        "test":   len(test_files),
        "unused": len(unused),
    }


def main() -> None:
    random.seed(SEED)

    print(f"Dividiendo dataset desde: {RAW_DIR}")
    print(f"Semilla reproducible:     SEED={SEED}")
    print(f"Límite por clase:         {LIMIT_PER_CLASS} (train={TRAIN_COUNT}, val={VAL_COUNT}, test={TEST_COUNT})\n")

    print(f"{'CLASE':<28} | {'TRAIN':>6} | {'VAL':>6} | {'TEST':>6} | {'UNUSED':>7}")
    print("-" * 65)

    for cls in CLASSES:
        result = split_class(cls)
        if result:
            print(
                f"{cls:<28} | {result['train']:>6} | {result['val']:>6} | "
                f"{result['test']:>6} | {result['unused']:>7}"
            )

    print("-" * 65)
    print("\nDataset listo. Siguiente paso: make train")


if __name__ == "__main__":
    main()
