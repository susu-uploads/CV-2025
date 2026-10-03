from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbformat.notebooknode import NotebookNode


PRACTICE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PRACTICE_DIR / "СП-М-О-КЗ-2025-Практика-5-1-БабушкинМВ.ipynb"


def md(text: str) -> NotebookNode:
    """Создает markdown-ячейку notebook из многострочной строки."""
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str) -> NotebookNode:
    """Создает code-ячейку notebook из многострочной строки."""
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md("# Практическая работа 5.1. Подготовка датасета"),
    md("## Импорты"),
    code(
        """
        from __future__ import annotations

        import os
        import shutil
        from pathlib import Path

        PRACTICE_DIR = Path.cwd()
        CACHE_DIR = PRACTICE_DIR / ".cache"
        DATA_DIR = PRACTICE_DIR / ".data"
        MPL_CACHE_DIR = CACHE_DIR / "matplotlib"
        IPYTHON_CACHE_DIR = CACHE_DIR / "ipython"
        XDG_CACHE_DIR = CACHE_DIR / "xdg"
        SOURCE_DATASET_DIR = DATA_DIR / "flower-classification-10-classes-source"
        PREPARED_DATASET_DIR = DATA_DIR / "flower-classification-7-classes-prepared"

        for directory in (CACHE_DIR, DATA_DIR, MPL_CACHE_DIR, IPYTHON_CACHE_DIR, XDG_CACHE_DIR):
            directory.mkdir(parents=True, exist_ok=True)

        os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)
        os.environ["IPYTHONDIR"] = str(IPYTHON_CACHE_DIR)
        os.environ["XDG_CACHE_HOME"] = str(XDG_CACHE_DIR)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        """
    ),
    code(
        """
        import kagglehub
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image

        IMAGE_SIZE = 180
        DATASET_HANDLE = "utkarshsaxenadn/flower-classification-5-classes-roselilyetc"
        DATASET_ROOT = SOURCE_DATASET_DIR / "Flower Classification V2" / "V2"
        SPLIT_DIRS = {
            "train": DATASET_ROOT / "Training Data",
            "validation": DATASET_ROOT / "Validation Data",
            "test": DATASET_ROOT / "Testing Data",
        }
        SELECTED_CLASSES = [
            ("Daisy", "Daisy"),
            ("Lavender", "Lavender"),
            ("Lily", "Lily"),
            ("Orchid", "Orchid"),
            ("Poppy", "Poppy"),
            ("Rose", "Rose"),
            ("Sunflower", "Sunflower"),
        ]

        plt.style.use("seaborn-v0_8-whitegrid")
        """
    ),
    md(
        """
        ## Описание данных

        В работе используется flower-датасет с Kaggle. Внутри набора есть версия `V2` на `10` классов. Для выполнения задания из нее выбираются `7` классов:

        - `Daisy`
        - `Lavender`
        - `Lily`
        - `Orchid`
        - `Poppy`
        - `Rose`
        - `Sunflower`

        Исходные выборки `train / validation / test` в датасете уже есть, поэтому в notebook выполняется только перепаковка выбранных классов в локальный prepared-набор.
        """
    ),
    md("## Загрузка датасета"),
    code(
        """
        def download_dataset_if_needed(
            source_dataset_dir: Path,
            dataset_root: Path,
            dataset_handle: str,
        ) -> None:
            \"\"\"
            Скачивает датасет Kaggle, если локальная копия еще не подготовлена.

            Parameters
            ----------
            source_dataset_dir : Path
                Каталог для скачивания датасета.
            dataset_root : Path
                Каталог с папками данных.
            dataset_handle : str
                Идентификатор датасета на Kaggle.
            \"\"\"
            if dataset_root.is_dir():
                return

            kagglehub.dataset_download(
                dataset_handle,
                output_dir=str(source_dataset_dir),
            )
        """
    ),
    code(
        """
        def collect_split_records(
            split_dirs: dict[str, Path],
            selected_classes: list[tuple[str, str]],
        ) -> tuple[dict[str, dict[str, list[Path]]], list[str]]:
            \"\"\"
            Собирает пути к изображениям выбранных классов по готовым split.

            Parameters
            ----------
            split_dirs : dict[str, Path]
                Пути к каталогам train, validation и test.
            selected_classes : list[tuple[str, str]]
                Пары из имени исходной папки и итогового имени класса.

            Returns
            -------
            tuple[dict[str, dict[str, list[Path]]], list[str]]
                Данные по split и итоговый список классов.
            \"\"\"
            image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            split_records: dict[str, dict[str, list[Path]]] = {}

            for split_name, split_dir in split_dirs.items():
                if not split_dir.is_dir():
                    raise FileNotFoundError(f"Не найдена папка split: {split_dir}")

                split_records[split_name] = {}

                for source_class_name, target_class_name in selected_classes:
                    class_dir = split_dir / source_class_name
                    if not class_dir.is_dir():
                        raise FileNotFoundError(f"Не найдена папка класса: {class_dir}")

                    image_paths = sorted(
                        path for path in class_dir.rglob("*")
                        if path.is_file() and path.suffix.lower() in image_suffixes
                    )

                    if not image_paths:
                        raise FileNotFoundError(f"Для класса {target_class_name} не найдены изображения.")

                    split_records[split_name][target_class_name] = image_paths

            class_names = [target_class_name for _, target_class_name in selected_classes]
            return split_records, class_names
        """
    ),
    code(
        """
        download_dataset_if_needed(
            SOURCE_DATASET_DIR,
            DATASET_ROOT,
            DATASET_HANDLE,
        )

        split_records, class_names = collect_split_records(
            SPLIT_DIRS,
            SELECTED_CLASSES,
        )

        split_class_counts = {
            split_name: {
                class_name: len(paths)
                for class_name, paths in class_records.items()
            }
            for split_name, class_records in split_records.items()
        }
        split_sizes = {
            split_name: sum(class_counts.values())
            for split_name, class_counts in split_class_counts.items()
        }

        print("Классы:", class_names)
        for split_name, class_counts in split_class_counts.items():
            print(f"\\n{split_name}:")
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count}")
            print(f"  total: {split_sizes[split_name]}")
        """
    ),
    md("## Обзор данных"),
    code(
        """
        fig, axis = plt.subplots(figsize=(5, 5))
        axis.pie(
            list(split_sizes.values()),
            labels=[f"{name}\\n{size}" for name, size in split_sizes.items()],
            startangle=90,
            counterclock=False,
        )
        axis.set_title("Общее разбиение датасета")
        axis.axis("equal")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        def plot_class_pie(counts: dict[str, int], title: str) -> None:
            \"\"\"
            Строит круговую диаграмму распределения изображений по классам.

            Parameters
            ----------
            counts : dict[str, int]
                Количество изображений по классам.
            title : str
                Заголовок диаграммы.
            \"\"\"
            fig, axis = plt.subplots(figsize=(6, 6))
            axis.pie(
                list(counts.values()),
                labels=[f"{name}\\n{count}" for name, count in counts.items()],
                startangle=90,
                counterclock=False,
            )
            axis.set_title(title)
            axis.axis("equal")
            plt.tight_layout()
            plt.show()
        """
    ),
    code('plot_class_pie(split_class_counts["train"], "Train: распределение по классам")'),
    code('plot_class_pie(split_class_counts["validation"], "Validation: распределение по классам")'),
    code('plot_class_pie(split_class_counts["test"], "Test: распределение по классам")'),
    md("## Превью изображений"),
    code(
        """
        preview_records = []

        for class_name in class_names:
            first_image = split_records["train"][class_name][0]
            preview_records.append((class_name, first_image))

        fig, axes = plt.subplots(3, 3, figsize=(10, 8))
        axes = np.asarray(axes).reshape(-1)

        for axis, (class_name, image_path) in zip(axes, preview_records):
            image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            axis.imshow(image)
            axis.set_title(class_name, fontsize=9)
            axis.axis("off")

        for axis in axes[len(preview_records):]:
            axis.axis("off")

        plt.tight_layout()
        plt.show()
        """
    ),
    md("## Сохранение подготовленного датасета"),
    code(
        """
        def create_prepared_dataset(
            prepared_dir: Path,
            split_records: dict[str, dict[str, list[Path]]],
        ) -> None:
            \"\"\"
            Копирует изображения в структуру train / validation / test для TensorFlow.

            Parameters
            ----------
            prepared_dir : Path
                Каталог для prepared-датасета.
            split_records : dict[str, dict[str, list[Path]]]
                Пути к изображениям по split и по классам.
            \"\"\"
            if prepared_dir.exists():
                shutil.rmtree(prepared_dir)

            for split_name, class_records in split_records.items():
                for class_name, paths in class_records.items():
                    target_dir = prepared_dir / split_name / class_name
                    target_dir.mkdir(parents=True, exist_ok=True)

                    for index, source_path in enumerate(paths, start=1):
                        target_name = f"{class_name}_{source_path.stem}_{index:05d}.jpg"
                        target_path = target_dir / target_name

                        image = Image.open(source_path).convert("RGB")
                        image.save(target_path, format="JPEG", quality=95)
        """
    ),
    code(
        """
        create_prepared_dataset(PREPARED_DATASET_DIR, split_records)
        print(PREPARED_DATASET_DIR)
        """
    ),
    code(
        """
        for split_name in ("train", "validation", "test"):
            split_dir = PREPARED_DATASET_DIR / split_name
            total = 0
            print(f"\\n{split_name}:")
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                count = len(list(class_dir.rglob("*.*")))
                total += count
                print(f"  {class_dir.name}: {count}")
            print(f"  total: {total}")
        """
    ),
]


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3.12 (.venv)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.12",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
        "file_extension": ".py",
    },
}

nbf.write(nb, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
