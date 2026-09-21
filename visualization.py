"""Dessin des bounding boxes et préparation du tableau de résultats."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from config import CLASS_COLORS


def draw_detections(
    image: Image.Image, detections: list[dict[str, Any]]
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    image_width, image_height = canvas.size
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    thickness = max(2, round(min(image_width, image_height) / 300))

    for detection in detections:
        class_name = detection["class_name"]
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        x1 = int(round(detection["x1"]))
        y1 = int(round(detection["y1"]))
        x2 = int(round(detection["x2"]))
        y2 = int(round(detection["y2"]))
        label = f"{class_name} {detection['confidence'] * 100:.1f}%"

        draw.rectangle((x1, y1, x2, y2), outline=color, width=thickness)
        text_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_top = max(0, y1 - text_height - 10)
        label_right = min(image_width - 1, x1 + text_width + 10)
        draw.rectangle((x1, label_top, label_right, y1), fill=color)
        draw.text(
            (x1 + 5, label_top + 3),
            label,
            fill=(12, 18, 28),
            font=font,
            stroke_width=1,
        )
    return canvas


def detections_dataframe(
    detections: list[dict[str, Any]], small_ratio: float
) -> pd.DataFrame:
    rows = []
    for index, detection in enumerate(detections, start=1):
        rows.append(
            {
                "Detection ID": index,
                "Class": detection["class_name"],
                "Confidence": detection["confidence"],
                "X1": round(detection["x1"], 1),
                "Y1": round(detection["y1"], 1),
                "X2": round(detection["x2"], 1),
                "Y2": round(detection["y2"], 1),
                "Width": round(detection["width"], 1),
                "Height": round(detection["height"], 1),
                "Image area (%)": round(detection["area_ratio"] * 100, 3),
                "Small object": detection["area_ratio"] < small_ratio,
            }
        )
    return pd.DataFrame(rows)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
