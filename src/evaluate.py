import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    BATCH_SIZE, CLASSES, DATA_DIR, MODEL_PATH,
    NUM_CLASSES, OUTPUTS_DIR, SEED,
)
from src.data.transforms import get_transforms
from src.model.resnet import load_model


def evaluate(model_path: str) -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    data_transforms = get_transforms()
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), data_transforms['test'])
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        worker_init_fn=lambda wid: np.random.seed(SEED + wid),
    )
    class_names = test_dataset.classes

    model = load_model(model_path, NUM_CLASSES, device)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\n" + "=" * 50)
    print("         REPORTE DE EVALUACIÓN FINAL")
    print("=" * 50)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    accuracy = np.mean(all_preds == all_labels) * 100
    print(f"Accuracy en test: {accuracy:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión — Dataset de Prueba')
    plt.ylabel('Clase Real')
    plt.xlabel('Clase Predicha')
    plt.tight_layout()

    output_path = os.path.join(OUTPUTS_DIR, 'confusion_matrix.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Matriz de confusión guardada en: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación del modelo — maíz")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Ruta al archivo .pth del modelo")
    args = parser.parse_args()
    evaluate(args.model)


if __name__ == "__main__":
    main()
