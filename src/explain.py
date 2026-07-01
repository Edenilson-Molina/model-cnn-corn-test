import argparse
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from lime import lime_image
from PIL import Image
from skimage.segmentation import mark_boundaries

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CLASSES, DATA_DIR, MODEL_PATH, NUM_CLASSES, OUTPUTS_DIR, SEED
from src.data.transforms import get_transforms
from src.model.resnet import load_model

EXPLANATIONS_DIR = os.path.join(OUTPUTS_DIR, "explanations")

CLASS_LABELS = {
    "gray_leaf_spot":       "Mancha Gris",
    "healthy":              "Hoja Sana",
    "magnesium_deficiency": "Deficiencia de Magnesio",
}

CLASS_COLORS = {
    "gray_leaf_spot":       "#E74C3C",
    "healthy":              "#27AE60",
    "magnesium_deficiency": "#F39C12",
}


def _batch_predict(model, device, transform):
    """Genera una función de predicción compatible con LIME (recibe numpy HWC)."""
    def predict_fn(images: np.ndarray) -> np.ndarray:
        batch = torch.stack([transform(Image.fromarray(img)) for img in images]).to(device)
        with torch.no_grad():
            outputs = model(batch)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        return probs.cpu().numpy()
    return predict_fn


def explain_image(
    image_path: str,
    model: torch.nn.Module,
    explainer: lime_image.LimeImageExplainer,
    predict_fn,
    transform,
    device: torch.device,
    num_samples: int = 300,
) -> str:
    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image.resize((224, 224)))

    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_idx = outputs.argmax(1).item()

    confidence = probs[0][pred_idx].item() * 100
    pred_class = CLASSES[pred_idx]
    label = CLASS_LABELS.get(pred_class, pred_class)
    color = CLASS_COLORS.get(pred_class, "#3498DB")
    print(f"  {Path(image_path).name} -> {label} ({confidence:.1f}%)")

    explanation = explainer.explain_instance(
        img_np,
        predict_fn,
        top_labels=NUM_CLASSES,
        hide_color=0,
        num_samples=num_samples,
    )

    # --- Visualización elegante ---
    fig = plt.figure(figsize=(16, 6), facecolor="white")
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.08], wspace=0.15)

    # Panel 1: Imagen original
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img_np)
    ax0.set_title("Imagen Original", fontsize=13, fontweight="bold", color="#2C3E50", pad=12)
    ax0.axis("off")

    # Panel 2: Regiones clave (superpíxeles positivos)
    ax1 = fig.add_subplot(gs[1])
    temp, mask = explanation.get_image_and_mask(
        pred_idx,
        positive_only=True,
        num_features=8,
        hide_rest=False,
    )
    overlay = mark_boundaries(temp / 255.0 if temp.max() > 1 else temp, mask, color=(0.1, 0.8, 0.3), mode="thick")
    # Resaltar las regiones importantes con tinte verde suave
    highlight = overlay.copy()
    highlight[mask == 1] = highlight[mask == 1] * 0.5 + np.array([0.2, 0.8, 0.3]) * 0.5
    ax1.imshow(np.clip(highlight, 0, 1))
    ax1.set_title("Zonas que Determinan\nel Diagnóstico", fontsize=13, fontweight="bold",
                  color="#2C3E50", pad=12)
    ax1.axis("off")

    # Panel 3: Mapa de calor de importancia
    ax2 = fig.add_subplot(gs[2])
    cax = fig.add_subplot(gs[3])

    ind = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.zeros(explanation.segments.shape, dtype=np.float64)
    for seg_id, weight in dict_heatmap.items():
        heatmap[explanation.segments == seg_id] = weight

    ax2.imshow(img_np, alpha=0.35)
    im = ax2.imshow(heatmap, cmap="RdYlGn", alpha=0.65, interpolation="bilinear")
    ax2.set_title("Mapa de Importancia", fontsize=13, fontweight="bold", color="#2C3E50", pad=12)
    ax2.axis("off")

    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=9)
    cb.set_label("Importancia", fontsize=10, color="#7F8C8D")

    # Título principal con diagnóstico
    fig.suptitle(
        f"Diagnóstico: {label}   —   Confianza: {confidence:.1f}%",
        fontsize=16, fontweight="bold", color=color, y=0.98,
    )

    # Pie de página
    fig.text(
        0.5, 0.01,
        "Las zonas verdes son las regiones de la hoja que el modelo consideró más importantes para su decisión.",
        ha="center", fontsize=10, color="#95A5A6", style="italic",
    )

    os.makedirs(EXPLANATIONS_DIR, exist_ok=True)
    stem = Path(image_path).stem
    output_path = os.path.join(EXPLANATIONS_DIR, f"{stem}_explicacion.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path


def sample_test_images(n: int) -> list:
    random.seed(SEED)
    test_dir = os.path.join(DATA_DIR, "test")
    paths = []
    for cls in CLASSES:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = sorted([
            os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        if files:
            paths.append(random.choice(files))
    extra = n - len(paths)
    if extra > 0:
        all_files = [
            os.path.join(root, f)
            for cls in CLASSES
            for root, _, files in os.walk(os.path.join(test_dir, cls))
            for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(all_files)
        paths += all_files[:extra]
    return paths[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicabilidad LIME — maíz")
    parser.add_argument("--image",       type=str, default=None)
    parser.add_argument("--n-images",    type=int, default=3)
    parser.add_argument("--model",       type=str, default=MODEL_PATH)
    parser.add_argument("--num-samples", type=int, default=300,
                        help="Perturbaciones LIME (mas = mejor pero mas lento)")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"Error: modelo no encontrado en '{args.model}'. Ejecuta 'make train' primero.")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    model = load_model(args.model, NUM_CLASSES, device)
    transforms = get_transforms()

    predict_fn = _batch_predict(model, device, transforms["val"])
    explainer = lime_image.LimeImageExplainer(random_state=SEED)

    images_to_explain = [args.image] if args.image else sample_test_images(args.n_images)

    print(f"\nGenerando explicaciones para {len(images_to_explain)} imagen(es):\n")
    saved = []
    for img_path in images_to_explain:
        out = explain_image(img_path, model, explainer, predict_fn, transforms["val"], device, args.num_samples)
        saved.append(out)

    print(f"\nGuardadas en: {EXPLANATIONS_DIR}/")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
