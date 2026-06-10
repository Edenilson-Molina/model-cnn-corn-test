"""
Script para detectar y eliminar imágenes corruptas del dataset.

El error 'UnidentifiedImageError' ocurre cuando PIL/Pillow no puede
leer un archivo como imagen válida (archivo truncado, corrupto, o
con extensión incorrecta).

Este script escanea train/, val/ y test/ buscando esos archivos.
"""

import os
from pathlib import Path
from PIL import Image

# Ruta base del dataset
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def find_corrupted_images(data_dir: Path) -> list[Path]:
    """Escanea todas las imágenes y retorna las que están corruptas."""
    corrupted = []
    total_checked = 0

    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"  [SKIP] Carpeta '{split}' no encontrada.")
            continue

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue

                total_checked += 1
                try:
                    with Image.open(img_path) as img:
                        img.verify()  # Verifica que el archivo sea una imagen válida
                except Exception as e:
                    corrupted.append(img_path)
                    print(f"  [CORRUPTA] {img_path.relative_to(data_dir)}  ->  {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"  Total de imágenes revisadas: {total_checked}")
    print(f"  Imágenes corruptas encontradas: {len(corrupted)}")
    print(f"{'='*60}")
    return corrupted


def delete_corrupted(corrupted: list[Path]):
    """Elimina las imágenes corruptas."""
    for path in corrupted:
        path.unlink()
        print(f"  [ELIMINADA] {path.name}")
    print(f"\n  Se eliminaron {len(corrupted)} archivo(s).")


if __name__ == "__main__":
    print(f"Escaneando dataset en: {DATA_DIR}\n")
    corrupted = find_corrupted_images(DATA_DIR)

    if not corrupted:
        print("\n¡No se encontraron imágenes corruptas! El dataset está limpio.")
    else:
        print(f"\nSe encontraron {len(corrupted)} imagen(es) corrupta(s):")
        for p in corrupted:
            print(f"  - {p.relative_to(DATA_DIR)}")

        respuesta = input("\n¿Deseas eliminarlas? (s/n): ").strip().lower()
        if respuesta == "s":
            delete_corrupted(corrupted)
            print("\nListo. Ahora puedes volver a ejecutar el entrenamiento.")
        else:
            print("\nNo se eliminó nada. Puedes revisarlas manualmente.")
