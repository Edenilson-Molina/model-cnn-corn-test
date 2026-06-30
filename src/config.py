import os

SEED = 42

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH  = os.path.join(MODELS_DIR, "resnet18_corn_model.pth")

BATCH_SIZE  = 4
NUM_EPOCHS  = 15
LR          = 0.001
NUM_CLASSES = 3
IMG_SIZE    = 224

CLASSES = ['gray_leaf_spot', 'healthy', 'magnesium_deficiency']
MEAN    = [0.485, 0.456, 0.406]
STD     = [0.229, 0.224, 0.225]
