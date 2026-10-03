from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbformat.notebooknode import NotebookNode


PRACTICE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PRACTICE_DIR / "СП-М-О-КЗ-2025-Практика-6-1-БабушкинМВ.ipynb"


def md(text: str) -> NotebookNode:
    """Создает markdown-ячейку notebook из многострочной строки."""
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str) -> NotebookNode:
    """Создает code-ячейку notebook из многострочной строки."""
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md(
        """
        # Практическая работа 6.1. Подготовка датасета

        В notebook скачивается датасет мебели, показывается его состав и формируется prepared-набор `train / validation / test` для дальнейшей работы в PyTorch.
        """
    ),
    md("## Импорты"),
    code(
        """
        from __future__ import annotations

        import os
        import random
        import shutil
        from pathlib import Path

        import kagglehub
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image

        PRACTICE_DIR = Path.cwd()
        CACHE_DIR = PRACTICE_DIR / ".cache"
        DATA_DIR = PRACTICE_DIR / ".data"
        MPL_CACHE_DIR = CACHE_DIR / "matplotlib"

        for directory in (CACHE_DIR, DATA_DIR, MPL_CACHE_DIR):
            directory.mkdir(parents=True, exist_ok=True)

        os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)

        SEED = 42
        TRAIN_FRACTION = 0.70
        VALIDATION_FRACTION = 0.15
        TEST_FRACTION = 0.15
        DATASET_HANDLE = "udaysankarmukherjee/furniture-image-dataset"
        SOURCE_DATASET_DIR = DATA_DIR / "furniture-image-dataset"
        PREPARED_DATASET_DIR = DATA_DIR / "furniture-image-dataset-prepared"
        CLASS_FOLDERS = {
            "almirah": "almirah_dataset",
            "chair": "chair_dataset",
            "fridge": "fridge dataset",
            "table": "table dataset",
            "television": "tv dataset",
        }

        if round(TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION, 5) != 1.0:
            raise ValueError("Сумма долей train, validation и test должна быть равна 1.")

        random.seed(SEED)
        np.random.seed(SEED)
        plt.style.use("seaborn-v0_8-whitegrid")
        """
    ),
    md(
        """
        ## Описание данных

        В работе используется `Furniture Image Dataset` с Kaggle. В датасете выбраны пять классов:

        - `almirah`
        - `chair`
        - `fridge`
        - `table`
        - `television`

        Исходный набор не имеет готового разбиения `train / validation / test`, поэтому оно формируется отдельно для каждого класса.
        """
    ),
    md("## Загрузка датасета"),
    code(
        """
        def load_furniture_dataset(
            dataset_dir: Path,
            dataset_handle: str,
            class_folders: dict[str, str],
        ) -> tuple[dict[str, list[Path]], list[str]]:
            \"\"\"
            Скачивает датасет и собирает изображения по папкам классов.

            Parameters
            ----------
            dataset_dir : Path
                Каталог, в который скачивается датасет.
            dataset_handle : str
                Идентификатор датасета Kaggle.
            class_folders : dict[str, str]
                Соответствие между названием класса и папкой внутри датасета.

            Returns
            -------
            tuple[dict[str, list[Path]], list[str]]
                Пути к изображениям по классам и список названий классов.
            \"\"\"
            kagglehub.dataset_download(dataset_handle, output_dir=str(dataset_dir))

            class_to_paths: dict[str, list[Path]] = {}
            class_names = list(class_folders.keys())

            for class_name in class_names:
                class_dir = dataset_dir / class_folders[class_name]
                image_paths = sorted(class_dir.glob("*.jpeg"))
                if not image_paths:
                    raise FileNotFoundError(f"В папке {class_dir} не найдены изображения.")
                class_to_paths[class_name] = image_paths

            return class_to_paths, class_names
        """
    ),
    code(
        """
        class_to_paths, class_names = load_furniture_dataset(
            SOURCE_DATASET_DIR,
            DATASET_HANDLE,
            CLASS_FOLDERS,
        )

        class_counts = {
            class_name: len(class_to_paths[class_name])
            for class_name in class_names
        }

        print(f"Количество классов: {len(class_names)}")
        print(f"Общее количество изображений: {sum(class_counts.values())}")
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count}")
        """
    ),
    md("## Просмотр данных"),
    code(
        """
        def plot_class_counts(counts: dict[str, int], title: str) -> None:
            \"\"\"
            Строит столбчатую диаграмму количества изображений по классам.

            Parameters
            ----------
            counts : dict[str, int]
                Количество изображений по классам.
            title : str
                Заголовок диаграммы.
            \"\"\"
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(counts.keys(), counts.values(), color="#80b1d3")
            ax.set_title(title)
            ax.set_ylabel("Количество изображений")
            plt.tight_layout()
            plt.show()
        """
    ),
    code('plot_class_counts(class_counts, "Количество изображений по классам")'),
    code(
        """
        preview_paths = [class_to_paths[class_name][0] for class_name in class_names]

        fig, axes = plt.subplots(2, 3, figsize=(10, 7))
        axes = np.asarray(axes).reshape(-1)

        for axis, image_path in zip(axes, preview_paths):
            image = Image.open(image_path).convert("RGB")
            axis.imshow(np.asarray(image))
            axis.set_title(image_path.parent.name, fontsize=9)
            axis.axis("off")

        for axis in axes[len(preview_paths):]:
            axis.axis("off")

        plt.tight_layout()
        plt.show()
        """
    ),
    md("## Формирование train / validation / test"),
    code(
        """
        def make_train_validation_test_split(
            class_to_paths: dict[str, list[Path]],
            train_fraction: float,
            validation_fraction: float,
            seed: int,
        ) -> dict[str, dict[str, list[Path]]]:
            \"\"\"
            Делит изображения каждого класса на train, validation и test.

            Parameters
            ----------
            class_to_paths : dict[str, list[Path]]
                Пути к изображениям по классам.
            train_fraction : float
                Доля изображений для train.
            validation_fraction : float
                Доля изображений для validation.
            seed : int
                Значение для воспроизводимого перемешивания.

            Returns
            -------
            dict[str, dict[str, list[Path]]]
                Пути к изображениям по split и классам.
            \"\"\"
            rng = random.Random(seed)
            split_records = {
                "train": {},
                "validation": {},
                "test": {},
            }

            for class_name, paths in class_to_paths.items():
                shuffled_paths = list(paths)
                rng.shuffle(shuffled_paths)

                train_count = int(len(shuffled_paths) * train_fraction)
                validation_count = int(len(shuffled_paths) * validation_fraction)

                split_records["train"][class_name] = shuffled_paths[:train_count]
                split_records["validation"][class_name] = shuffled_paths[
                    train_count:train_count + validation_count
                ]
                split_records["test"][class_name] = shuffled_paths[
                    train_count + validation_count:
                ]

            return split_records
        """
    ),
    code(
        """
        split_records = make_train_validation_test_split(
            class_to_paths,
            TRAIN_FRACTION,
            VALIDATION_FRACTION,
            SEED,
        )

        split_counts = {
            split_name: {
                class_name: len(paths)
                for class_name, paths in class_records.items()
            }
            for split_name, class_records in split_records.items()
        }

        for split_name, counts in split_counts.items():
            print(f"\\n{split_name}: {sum(counts.values())}")
            for class_name, count in counts.items():
                print(f"  {class_name}: {count}")
        """
    ),
    code(
        """
        fig, axis = plt.subplots(figsize=(6, 6))
        axis.pie(
            [
                sum(split_counts["train"].values()),
                sum(split_counts["validation"].values()),
                sum(split_counts["test"].values()),
            ],
            labels=[
                f"train\\n{sum(split_counts['train'].values())}",
                f"validation\\n{sum(split_counts['validation'].values())}",
                f"test\\n{sum(split_counts['test'].values())}",
            ],
            startangle=90,
            counterclock=False,
        )
        axis.set_title("Разбиение датасета")
        axis.axis("equal")
        plt.tight_layout()
        plt.show()
        """
    ),
    md("## Создание prepared-датасета"),
    code(
        """
        def create_prepared_dataset(
            split_records: dict[str, dict[str, list[Path]]],
            prepared_dataset_dir: Path,
        ) -> None:
            \"\"\"
            Перепаковывает train, validation и test в локальный prepared-датасет.

            Parameters
            ----------
            split_records : dict[str, dict[str, list[Path]]]
                Пути к изображениям по split и классам.
            prepared_dataset_dir : Path
                Каталог prepared-датасета.
            \"\"\"
            if prepared_dataset_dir.exists():
                shutil.rmtree(prepared_dataset_dir)

            for split_name, class_records in split_records.items():
                for class_name, image_paths in class_records.items():
                    class_dir = prepared_dataset_dir / split_name / class_name
                    class_dir.mkdir(parents=True, exist_ok=True)

                    for image_path in image_paths:
                        shutil.copy2(image_path, class_dir / image_path.name)
        """
    ),
    code(
        """
        create_prepared_dataset(
            split_records,
            PREPARED_DATASET_DIR,
        )

        print(PREPARED_DATASET_DIR)
        """
    ),
    code(
        """
        for split_name in ("train", "validation", "test"):
            split_root = PREPARED_DATASET_DIR / split_name
            print(f"\\n{split_name}:")
            for class_name in class_names:
                class_count = len(list((split_root / class_name).glob("*.jpeg")))
                print(f"  {class_name}: {class_count}")
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
