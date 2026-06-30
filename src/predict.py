import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_PATH, NUM_CLASSES
from src.data.transforms import get_transforms
from src.model.resnet import load_model


def predict(image_path: str, model_path: str) -> tuple[str, float]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from src.config import CLASSES
    model = load_model(model_path, NUM_CLASSES, device)

    transform = get_transforms()['val']
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    predicted_class = CLASSES[predicted.item()]
    confidence = probabilities[0][predicted.item()].item() * 100

    print(f"Predicción : {predicted_class}")
    print(f"Confianza  : {confidence:.2f}%")
    return predicted_class, confidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia sobre imagen de maíz")
    parser.add_argument("--image", type=str, required=True,
                        help="Ruta a la imagen a clasificar")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Ruta al archivo .pth del modelo")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: imagen no encontrada en '{args.image}'")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"Error: modelo no encontrado en '{args.model}'. Ejecuta 'make train' primero.")
        sys.exit(1)

    predict(args.image, args.model)


if __name__ == "__main__":
    main()
