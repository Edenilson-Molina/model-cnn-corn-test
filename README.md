# 🌽 Clasificador de Enfermedades en Maíz con CNN

Modelo de **Red Neuronal Convolucional (CNN)** basado en **ResNet-18** para la clasificación automática de enfermedades y deficiencias nutricionales en hojas de maíz a partir de imágenes.

## 🎯 Objetivo

Desarrollar un sistema de visión por computadora capaz de identificar el estado de salud de plantas de maíz analizando fotografías de sus hojas, clasificándolas en **3 categorías activas**:

| Clase | Descripción |
|-------|-------------|
| `gray_leaf_spot` | Mancha gris de la hoja (enfermedad fúngica) |
| `healthy` | Hoja sana |
| `magnesium_deficiency` | Deficiencia de magnesio |


## 🧠 Arquitectura del Modelo

- **Modelo base:** ResNet-18 preentrenada en ImageNet
- **Estrategia:** Transfer Learning con capas congeladas
- **Capa final:** Capa fully connected adaptada a 3 clases
- **Optimizador:** Adam (lr=0.001)
- **Función de pérdida:** CrossEntropyLoss
- **Épocas de entrenamiento:** 15
- **Tamaño de entrada:** 224×224 px

## 📊 Dataset

600 imágenes por clase, balanceadas y divididas en:

| Split | Proporción | Imágenes por clase |
|-------|:----------:|:------------------:|
| Train | 70% | 420 |
| Val | 15% | 90 |
| Test | 15% | 90 |

Las imágenes son fotografías reales de hojas de maíz en campo.

## 📁 Estructura del Proyecto

```
model-cnn-corn/
├── clean/                    # Imágenes originales sin procesar (9 clases)
├── data/                     # Dataset organizado para entrenamiento
│   ├── train/                #   └── {clase}/ (420 imgs c/u)
│   ├── val/                  #   └── {clase}/ (90 imgs c/u)
│   └── test/                 #   └── {clase}/ (90 imgs c/u)
├── script/                   # Scripts de utilidad
│   ├── organize_dataset.py                 # Split inicial del dataset (70/15/15)
│   ├── balance_and_split.py                # Balanceo a 600 imgs/clase + re-split
│   ├── clean_corrupted_images.py           # Detección y eliminación de imgs corruptas
│   └── move_gray_leaf_spot_cropdg_real.py  # Migración de imágenes específicas
├── src/
│   ├── notebooks/
│   │   ├── red_neuronal.ipynb        # Notebook principal (entrenamiento + evaluación)
│   │   └── resnet18_corn_disease.pth # Modelo entrenado (pesos)
│   └── utils/                        # Utilidades (vacío por ahora)
├── venv/                     # Entorno virtual de Python
└── .gitignore
```

## ⚙️ Requisitos

- **Python** 3.14+
- Dependencias principales:

| Paquete | Uso |
|---------|-----|
| `torch` + `torchvision` | Framework de deep learning y carga de datos |
| `scikit-learn` | Métricas de evaluación (classification report, confusion matrix) |
| `matplotlib` + `seaborn` | Visualización de resultados |
| `Pillow` | Procesamiento de imágenes |
| `numpy` + `pandas` | Manipulación numérica y de datos |

## 🚀 Inicio Rápido

```bash
# Clonar el repositorio
git clone https://github.com/Edenilson-Molina/model-cnn-corn-test.git

# Crear y activar el entorno virtual
python -m venv venv
.\venv\Scripts\activate    # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## 🔮 Uso — Predicción de una Imagen

Una vez entrenado el modelo, se puede clasificar cualquier imagen de hoja de maíz:

```python
clase, imagen = predict_image(
    image_path="ruta/a/imagen.jpg",
    model=model,
    data_transforms=data_transforms,
    class_names=class_names,
    device=device
)
print(f"Diagnóstico: {clase}")
```
