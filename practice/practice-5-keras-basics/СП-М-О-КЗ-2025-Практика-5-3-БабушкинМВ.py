from __future__ import annotations

import os
from pathlib import Path

PRACTICE_DIR = Path(__file__).resolve().parent
CACHE_DIR = PRACTICE_DIR / ".cache"
MPL_CACHE_DIR = CACHE_DIR / "matplotlib"
IPYTHON_CACHE_DIR = CACHE_DIR / "ipython"
XDG_CACHE_DIR = CACHE_DIR / "xdg"
PREPARED_DATASET_DIR = PRACTICE_DIR / ".data" / "wonders-of-world-prepared"
TRAIN_ROOT = PREPARED_DATASET_DIR / "train"
MODEL_PATH = PRACTICE_DIR / "СП-М-О-КЗ-2025-Практика-5-2-БабушкинМВ.keras"
IMAGE_SIZE = (128, 128)

for directory in (CACHE_DIR, MPL_CACHE_DIR, IPYTHON_CACHE_DIR, XDG_CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)
os.environ["IPYTHONDIR"] = str(IPYTHON_CACHE_DIR)
os.environ["XDG_CACHE_HOME"] = str(XDG_CACHE_DIR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image


def load_class_names(train_root: Path) -> list[str]:
    """
    Считывает названия классов из каталога train prepared-датасета.

    Parameters
    ----------
    train_root : Path
        Каталог train подготовленного датасета.

    Returns
    -------
    list[str]
        Отсортированный список названий классов.
    """
    if not train_root.is_dir():
        raise FileNotFoundError(
            f"Каталог классов не найден: {train_root}. Сначала выполните notebook 5.1."
        )

    class_names = sorted(
        directory.name
        for directory in train_root.iterdir()
        if directory.is_dir()
    )

    if not class_names:
        raise FileNotFoundError(f"В каталоге {train_root} не найдены классы.")

    return class_names


CLASS_NAMES = load_class_names(TRAIN_ROOT)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Файл модели не найден: {MODEL_PATH}. Сначала выполните notebook 5.2."
    )

model = tf.keras.models.load_model(MODEL_PATH)


def prepare_image(image: Image.Image) -> np.ndarray:
    """
    Подготавливает пользовательское изображение для входа модели.

    Parameters
    ----------
    image : Image.Image
        Изображение, загруженное через интерфейс Gradio.

    Returns
    -------
    np.ndarray
        Batch из одного RGB-изображения нужного размера.
    """
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def classify_image(image: Image.Image | None) -> dict[str, float]:
    """
    Классифицирует изображение и возвращает вероятности классов.

    Parameters
    ----------
    image : Image.Image | None
        Пользовательское изображение из Gradio.

    Returns
    -------
    dict[str, float]
        Словарь с вероятностями для каждого класса.
    """
    if image is None:
        return {}

    image_batch = prepare_image(image)
    probabilities = model.predict(image_batch, verbose=0)[0]

    return {
        class_name: float(probability)
        for class_name, probability in zip(CLASS_NAMES, probabilities, strict=True)
    }


def create_interface() -> gr.Interface:
    """
    Создает Gradio-интерфейс для классификации изображений.

    Returns
    -------
    gr.Interface
        Готовый интерфейс Gradio.
    """
    return gr.Interface(
        fn=classify_image,
        inputs=gr.Image(type="pil", label="Изображение"),
        outputs=gr.Label(num_top_classes=5, label="Предсказание"),
        title="Wonders of the World",
        description="Классификация изображений по 7 классам чудес света.",
        flagging_mode="never",
    )


if __name__ == "__main__":
    create_interface().launch()
