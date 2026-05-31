from __future__ import annotations

import os
from pathlib import Path

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


WINDOWS_TESSERACT_PATHS = [
    r"C:\Users\artwi\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


def set_tesseract_path(custom_path: str | None = None) -> str | None:
    paths = []
    if custom_path:
        paths.append(custom_path)
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path:
        paths.append(env_path)
    paths.extend(WINDOWS_TESSERACT_PATHS)
    for value in paths:
        path = Path(value)
        if path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(path)
            return str(path)
    return None


def prepare_image(image: Image.Image) -> Image.Image:
    result = image.convert("L")
    result = ImageEnhance.Contrast(result).enhance(1.9)
    result = result.filter(ImageFilter.SHARPEN)
    return result


def read_image_text(image: Image.Image, language: str = "eng", custom_path: str | None = None) -> str:
    set_tesseract_path(custom_path)
    prepared = prepare_image(image)
    text = pytesseract.image_to_string(prepared, lang=language)
    return " ".join(text.split())
