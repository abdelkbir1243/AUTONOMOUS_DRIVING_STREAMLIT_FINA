"""Configuration centralisée du démonstrateur Streamlit."""

from pathlib import Path


APP_TITLE = "Autonomous Driving Vision Lab"
APP_SUBTITLE = "JEPA-Based Object Detection for Autonomous Driving"

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "models" / "best_detector.pt"

INPUT_WIDTH = 1024
INPUT_HEIGHT = 320
PATCH_SIZE = 16
EMBED_DIM = 384

CLASS_NAMES = {
    0: "__background__",
    1: "Pedestrian",
    2: "Cyclist",
    3: "Car",
    4: "Van",
}

DISPLAY_CLASSES = ["Pedestrian", "Cyclist", "Car", "Van"]

# Couleurs RGB utilisées sur l'image annotée.
CLASS_COLORS = {
    "Pedestrian": (255, 92, 92),
    "Cyclist": (255, 193, 7),
    "Car": (44, 203, 139),
    "Van": (71, 152, 255),
}

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

DEFAULT_CONFIDENCE = 0.50
IOU_DISPLAY_VALUE = 0.50
DEFAULT_SMALL_OBJECT_RATIO = 0.01

# Performances mesurées sur les 1 497 images de validation KITTI,
# avec le meilleur checkpoint de l'époque 35.
VALIDATION_METRICS = {
    "Pedestrian": {
        "AP@50": 0.726110,
        "Recall E2": 0.799107,
        "Precision@0.50": 0.718468,
        "Recall@0.50": 0.712054,
        "F1@0.50": 0.715247,
    },
    "Cyclist": {
        "AP@50": 0.835158,
        "Recall E2": 0.869281,
        "Precision@0.50": 0.800000,
        "Recall@0.50": 0.797386,
        "F1@0.50": 0.798691,
    },
    "Car": {
        "AP@50": 0.932775,
        "Recall E2": 0.944014,
        "Precision@0.50": 0.887526,
        "Recall@0.50": 0.916901,
        "F1@0.50": 0.901974,
    },
    "Van": {
        "AP@50": 0.910020,
        "Recall E2": 0.936057,
        "Precision@0.50": 0.808989,
        "Recall@0.50": 0.895204,
        "F1@0.50": 0.849916,
    },
}

