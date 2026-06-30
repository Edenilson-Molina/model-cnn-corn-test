"""
Etapa 1 del pipeline de datos: clean/ → data/raw/

Escanea clean/{clase}/ recursivamente, valida cada imagen con PIL,
y copia las válidas a data/raw/{clase}/ con nombre de archivo aplanado.
Las imágenes corruptas se descartan automáticamente (sin prompt interactivo).
"""

import os
import shutil
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import BASE_DIR, CLASSES

CLEAN_DIR = Path(BASE_DIR) / "clean"
RAW_DIR   = Path(BASE_DIR) / "data" / "raw"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def collect_class(cls: str) -> dict:
    src_dir = CLEAN_DIR / cls
    dst_dir = RAW_DIR / cls

    if not src_dir.exists():
        print(f"  [SKIP] clean/{cls}/ no encontrada.")
        return {"found": 0, "copied": 0, "skipped": 0, "corrupted": 0}

    dst_dir.mkdir(parents=True, exist_ok=True)

    found = corrupted = skipped = copied = 0

    for img_path in sorted(src_dir.rglob("*")):
        if img_path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        found += 1

        if not is_valid_image(img_path):
            print(f"  [CORRUPTA] {img_path.relative_to(CLEAN_DIR)}")
            corrupted += 1
            continue

        dest = dst_dir / img_path.name
        if dest.exists():
            skipped += 1
            continue

        shutil.copy2(img_path, dest)
        copied += 1

    return {"found": found, "copied": copied, "skipped": skipped, "corrupted": corrupted}


def main() -> None:
    print(f"Recolectando imágenes desde: {CLEAN_DIR}")
    print(f"Destino:                     {RAW_DIR}\n")

    totals = {"found": 0, "copied": 0, "skipped": 0, "corrupted": 0}

    for cls in CLASSES:
        print(f"[{cls}]")
        result = collect_class(cls)
        for k in totals:
            totals[k] += result[k]
        print(
            f"  Encontradas: {result['found']}  |  "
            f"Copiadas: {result['copied']}  |  "
            f"Ya existían: {result['skipped']}  |  "
            f"Corruptas: {result['corrupted']}\n"
        )

    print("=" * 55)
    print(
        f"TOTAL  Encontradas: {totals['found']}  |  "
        f"Copiadas: {totals['copied']}  |  "
        f"Ya existían: {totals['skipped']}  |  "
        f"Corruptas: {totals['corrupted']}"
    )
    print("=" * 55)
    print(f"\nImágenes listas en: {RAW_DIR}")
    print("Siguiente paso: python script/split_data.py")


if __name__ == "__main__":
    main()
