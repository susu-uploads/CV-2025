from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbformat.notebooknode import NotebookNode


PRACTICE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PRACTICE_DIR / "СП-М-О-КЗ-2025-Практика-6-БабушкинМВ.ipynb"


def md(text: str) -> NotebookNode:
    """Создает markdown-ячейку notebook из многострочной строки."""
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str) -> NotebookNode:
    """Создает code-ячейку notebook из многострочной строки."""
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md(
        """
        # Практическая работа 6. Классификация изображений с использованием PyTorch

        В работе обучаются две сверточные модели PyTorch на датасете из пяти классов мебели: без аугментации и с `TrivialAugmentWide`.
        """
    ),
    md(
        """
        ## Импорты
        """
    ),
    code(
        """
        from __future__ import annotations

        import os
        from pathlib import Path

        PRACTICE_DIR = Path.cwd()
        CACHE_DIR = PRACTICE_DIR / ".cache"
        DATA_DIR = PRACTICE_DIR / ".data"
        MPL_CACHE_DIR = CACHE_DIR / "matplotlib"

        for directory in (CACHE_DIR, DATA_DIR, MPL_CACHE_DIR):
            directory.mkdir(parents=True, exist_ok=True)

        os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)
        """
    ),
    code(
        """
        import random
        from typing import Sequence

        import kagglehub
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        from PIL import Image
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
        from sklearn.model_selection import train_test_split
        from torch import nn
        from torch.utils.data import DataLoader, Dataset, Subset
        from torchinfo import summary
        from torchvision import transforms

        SEED = 42
        BATCH_SIZE = 64
        EPOCHS = 5
        IMAGE_SIZE = (128, 128)
        TRAIN_FRACTION = 0.70
        VALIDATION_FRACTION = 0.15
        NUM_WORKERS = 0

        DATASET_HANDLE = "udaysankarmukherjee/furniture-image-dataset"
        DATASET_DIR = DATA_DIR / "furniture-image-dataset"
        CLASS_FOLDERS = {
            "almirah": "almirah_dataset",
            "chair": "chair_dataset",
            "fridge": "fridge dataset",
            "table": "table dataset",
            "television": "tv dataset",
        }

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        plt.style.use("seaborn-v0_8-whitegrid")

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        print(f"PyTorch: {torch.__version__}")
        print(f"Устройство: {device}")
        """
    ),
    md(
        """
        ## Описание данных

        В работе используется `Furniture Image Dataset` с Kaggle. В нем ровно пять классов мебели: шкаф, стул, холодильник, стол и телевизор.
        """
    ),
    md(
        """
        ### Загрузка датасета
        """
    ),
    code(
        """
        def load_furniture_dataset(
            dataset_dir: Path,
            dataset_handle: str,
            class_folders: dict[str, str],
        ) -> tuple[list[Path], np.ndarray, list[str]]:
            \"\"\"
            Скачивает датасет и собирает изображения из известных папок классов.

            Parameters
            ----------
            dataset_dir : Path
                Каталог, куда скачивается датасет.
            dataset_handle : str
                Идентификатор датасета Kaggle.
            class_folders : dict[str, str]
                Соответствие между названием класса и папкой внутри датасета.

            Returns
            -------
            tuple[list[Path], np.ndarray, list[str]]
                Пути к изображениям, числовые метки и названия классов.
            \"\"\"
            kagglehub.dataset_download(
                dataset_handle,
                output_dir=str(dataset_dir),
            )

            image_paths: list[Path] = []
            labels: list[int] = []
            class_names = list(class_folders.keys())

            for label, class_name in enumerate(class_names):
                class_dir = dataset_dir / class_folders[class_name]
                if not class_dir.exists():
                    raise FileNotFoundError(f"Не найдена папка класса: {class_dir}")

                class_images = sorted(class_dir.glob("*.jpeg"))
                if not class_images:
                    raise FileNotFoundError(f"В папке {class_dir} не найдены изображения.")

                image_paths.extend(class_images)
                labels.extend([label] * len(class_images))

            return image_paths, np.asarray(labels, dtype=np.int64), class_names
        """
    ),
    code(
        """
        image_paths, targets, class_names = load_furniture_dataset(
            DATASET_DIR,
            DATASET_HANDLE,
            CLASS_FOLDERS,
        )
        full_class_counts = np.bincount(targets, minlength=len(class_names))

        print(f"Количество изображений: {len(image_paths)}")
        print(f"Количество классов: {len(class_names)}")
        print("Классы:", class_names)
        """
    ),
    code(
        """
        for class_name, count in zip(class_names, full_class_counts):
            print(f"{class_name}: {count}")
        """
    ),
    md(
        """
        ## Подготовка данных

        На этом шаге датасет делится на обучающую, валидационную и тестовую выборки. Разбиение делается стратифицированно: каждый класс делится отдельно, поэтому доли классов сохраняются во всех трех выборках.
        """
    ),
    code(
        """
        def make_train_validation_test_split(
            targets: Sequence[int],
            *,
            train_fraction: float,
            validation_fraction: float,
            seed: int,
        ) -> tuple[list[int], list[int], list[int]]:
            \"\"\"
            Делит индексы на train, validation и test с помощью train_test_split.

            Parameters
            ----------
            targets : Sequence[int]
                Числовые метки всех изображений.
            train_fraction : float
                Доля объектов для обучающей выборки.
            validation_fraction : float
                Доля объектов для валидационной выборки.
            seed : int
                Значение для воспроизводимого разбиения.

            Returns
            -------
            tuple[list[int], list[int], list[int]]
                Индексы объектов для train, validation и test.
            \"\"\"
            all_indices = np.arange(len(targets))

            train_indices, temp_indices, _, temp_targets = train_test_split(
                all_indices,
                targets,
                test_size=1 - train_fraction,
                random_state=seed,
                stratify=targets,
            )

            validation_share_in_temp = validation_fraction / (1 - train_fraction)
            validation_indices, test_indices = train_test_split(
                temp_indices,
                train_size=validation_share_in_temp,
                random_state=seed,
                stratify=temp_targets,
            )

            return (
                train_indices.tolist(),
                validation_indices.tolist(),
                test_indices.tolist(),
            )
        """
    ),
    code(
        """
        train_indices, validation_indices, test_indices = make_train_validation_test_split(
            targets,
            train_fraction=TRAIN_FRACTION,
            validation_fraction=VALIDATION_FRACTION,
            seed=SEED,
        )

        train_class_counts = np.bincount(targets[train_indices], minlength=len(class_names))
        validation_class_counts = np.bincount(targets[validation_indices], minlength=len(class_names))
        test_class_counts = np.bincount(targets[test_indices], minlength=len(class_names))

        preview_indices = train_indices[:12]

        print(f"Train: {len(train_indices)}")
        print(f"Validation: {len(validation_indices)}")
        print(f"Test: {len(test_indices)}")
        """
    ),
    md(
        """
        ### Распределение данных

        Диаграммы показывают, сколько изображений попало в каждую выборку и как внутри выборок распределены классы.
        """
    ),
    code(
        """
        split_sizes = [len(train_indices), len(validation_indices), len(test_indices)]
        split_labels = [
            f"Train\\n{split_sizes[0]}",
            f"Validation\\n{split_sizes[1]}",
            f"Test\\n{split_sizes[2]}",
        ]

        fig, axis = plt.subplots(figsize=(5, 5))
        axis.pie(
            split_sizes,
            labels=split_labels,
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
        def plot_class_pie(counts: np.ndarray, title: str) -> None:
            \"\"\"
            Строит круговую диаграмму распределения изображений по классам.

            Parameters
            ----------
            counts : np.ndarray
                Количество изображений каждого класса.
            title : str
                Заголовок диаграммы.
            \"\"\"
            labels = [
                f"{class_name}\\n{count}"
                for class_name, count in zip(class_names, counts)
            ]

            fig, axis = plt.subplots(figsize=(5, 5))
            axis.pie(
                counts,
                labels=labels,
                startangle=90,
                counterclock=False,
            )
            axis.set_title(title)
            axis.axis("equal")
            plt.tight_layout()
            plt.show()
        """
    ),
    code('plot_class_pie(train_class_counts, "Train: распределение по классам")'),
    code('plot_class_pie(validation_class_counts, "Validation: распределение по классам")'),
    code('plot_class_pie(test_class_counts, "Test: распределение по классам")'),
    md(
        """
        ## Dataset и DataLoader

        Здесь задается общий класс датасета и общий способ сборки `DataLoader`. Сами преобразования изображений будут задаваться отдельно в каждом эксперименте.
        """
    ),
    code(
        """
        class FurnitureDataset(Dataset):
            \"\"\"
            Простой Dataset для изображений мебели.
            \"\"\"

            def __init__(
                self,
                image_paths: Sequence[Path],
                labels: Sequence[int],
                transform: transforms.Compose | None = None,
            ) -> None:
                \"\"\"
                Сохраняет пути к изображениям, метки и преобразования.

                Parameters
                ----------
                image_paths : Sequence[Path]
                    Пути к изображениям.
                labels : Sequence[int]
                    Числовые метки изображений.
                transform : transforms.Compose | None
                    Преобразования, применяемые к изображению.
                \"\"\"
                self.image_paths = list(image_paths)
                self.labels = list(labels)
                self.transform = transform

            def __len__(self) -> int:
                \"\"\"
                Возвращает количество изображений в датасете.
                \"\"\"
                return len(self.image_paths)

            def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
                \"\"\"
                Загружает изображение и возвращает его вместе с меткой.
                \"\"\"
                image = Image.open(self.image_paths[index]).convert("RGB")
                label = int(self.labels[index])

                if self.transform is not None:
                    image = self.transform(image)

                return image, label
        """
    ),
    code(
        """
        def make_dataloaders(
            train_transform: transforms.Compose,
            eval_transform: transforms.Compose,
        ) -> tuple[DataLoader, DataLoader, DataLoader]:
            \"\"\"
            Создает DataLoader для train, validation и test выборок.

            Parameters
            ----------
            train_transform : transforms.Compose
                Преобразования для обучающих изображений.
            eval_transform : transforms.Compose
                Преобразования для validation и test изображений.

            Returns
            -------
            tuple[DataLoader, DataLoader, DataLoader]
                DataLoader для train, validation и test.
            \"\"\"
            # Train получает свои преобразования, а validation и test остаются без случайной аугментации.
            train_dataset_full = FurnitureDataset(image_paths, targets, transform=train_transform)
            eval_dataset_full = FurnitureDataset(image_paths, targets, transform=eval_transform)

            # Subset использует уже подготовленные индексы стратифицированного разбиения.
            train_dataset = Subset(train_dataset_full, train_indices)
            validation_dataset = Subset(eval_dataset_full, validation_indices)
            test_dataset = Subset(eval_dataset_full, test_indices)

            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
            )

            return train_loader, validation_loader, test_loader
        """
    ),
    md(
        """
        ## Модель

        Модель построена по схеме из методического примера. Она состоит из двух сверточных блоков `Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d` и простого классификатора `Flatten -> Linear`.

        Первый блок выделяет простые локальные признаки изображения, а второй работает уже с более сложными комбинациями этих признаков. `ReLU` добавляет нелинейность, а `MaxPool2d` уменьшает пространственный размер карт признаков.

        После двух сверточных блоков карты признаков разворачиваются в вектор, и один полносвязный слой сразу выдает logits для пяти классов. Схема модели повторяет методический пример, а размер последнего линейного слоя подстраивается под текущий размер изображения и число классов.
        """
    ),
    code(
        """
        class FurnitureCNN(nn.Module):
            \"\"\"
            Сверточная нейронная сеть для классификации пяти видов мебели.
            \"\"\"

            def __init__(
                self,
                input_shape: int,
                hidden_units: int,
                output_shape: int,
            ) -> None:
                \"\"\"
                Создает сверточные блоки и классификатор.

                Parameters
                ----------
                input_shape : int
                    Количество входных каналов изображения.
                hidden_units : int
                    Количество каналов в сверточных слоях.
                output_shape : int
                    Количество классов в задаче.
                \"\"\"
                super().__init__()

                self.conv_block_1 = nn.Sequential(
                    # Conv2d - сверточный слой.
                    # in_channels задает число каналов на входе: для RGB это 3.
                    # out_channels задает число карт признаков, которые слой должен построить.
                    # kernel_size=3 означает окно свертки 3x3.
                    # stride=1 означает сдвиг окна на 1 пиксель.
                    # padding=1 добавляет по 1 пикселю рамки с каждой стороны и сохраняет размер изображения.
                    nn.Conv2d(
                        in_channels=input_shape,
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    # ReLU оставляет положительные значения и заменяет отрицательные на 0.
                    nn.ReLU(),
                    # Второй Conv2d берет карты признаков предыдущего слоя и уточняет их.
                    # Здесь число входных и выходных каналов одинаковое, потому что блок работает в одном пространстве признаков.
                    nn.Conv2d(
                        in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    # Повторная нелинейность после второй свертки.
                    nn.ReLU(),
                    # MaxPool2d уменьшает ширину и высоту в 2 раза.
                    # kernel_size=2 означает, что из каждого окна 2x2 берется максимальное значение.
                    nn.MaxPool2d(kernel_size=2),
                )

                self.conv_block_2 = nn.Sequential(
                    # Третий Conv2d начинает второй сверточный блок и работает с уже найденными признаками.
                    nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
                    # Нелинейность после свертки.
                    nn.ReLU(),
                    # Четвертый Conv2d еще раз уточняет признаки перед уменьшением размерности.
                    nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
                    # Нелинейность после второй свертки блока.
                    nn.ReLU(),
                    # Второй pooling снова уменьшает пространственный размер в 2 раза.
                    nn.MaxPool2d(kernel_size=2),
                )

                self.classifier = nn.Sequential(
                    # Flatten разворачивает многомерные карты признаков в один длинный вектор.
                    nn.Flatten(),
                    # Linear - полносвязный слой классификации.
                    # in_features - длина вектора после двух операций MaxPool2d.
                    # output_shape - количество итоговых классов.
                    nn.Linear(
                        hidden_units * (IMAGE_SIZE[0] // 4) * (IMAGE_SIZE[1] // 4),
                        output_shape,
                    ),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                \"\"\"
                Выполняет прямой проход изображения через модель.

                Parameters
                ----------
                x : torch.Tensor
                    Batch изображений формы (N, 3, H, W).

                Returns
                -------
                torch.Tensor
                    Logits для каждого класса.
                \"\"\"
                # Первый блок извлекает базовые признаки изображения.
                x = self.conv_block_1(x)
                # Второй блок уточняет и укрупняет найденные признаки.
                x = self.conv_block_2(x)
                # Классификатор переводит признаки в logits классов.
                return self.classifier(x)
        """
    ),
    code(
        """
        model_for_summary = FurnitureCNN(
            input_shape=3,
            hidden_units=10,
            output_shape=len(class_names),
        ).to(device)
        loss_fn = nn.CrossEntropyLoss()

        model_for_summary
        """
    ),
    md(
        """
        ### Проверка структуры модели

        `torchinfo.summary` показывает размеры тензоров после слоев и количество обучаемых параметров модели.
        """
    ),
    code(
        """
        summary(model_for_summary, input_size=(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]), device=device)
        """
    ),
    md(
        """
        ## Обучение модели

        В этом разделе задается общий пайплайн обучения. Функция `train_step` получает модель, `DataLoader` обучающей выборки, функцию потерь, оптимизатор и устройство, выполняет одну эпоху и возвращает средние `loss` и `accuracy` на train.

        Функция `test_step` принимает те же основные объекты, но работает без обновления весов и используется для validation или test. Функция `train` по эпохам вызывает `train_step` и `test_step`, сохраняет историю метрик и подготавливает данные для последующих графиков.
        """
    ),
    code(
        """
        def train_step(
            model: nn.Module,
            dataloader: DataLoader,
            loss_fn: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
        ) -> tuple[float, float]:
            \"\"\"
            Выполняет одну эпоху обучения модели.

            Parameters
            ----------
            model : nn.Module
                Обучаемая модель.
            dataloader : DataLoader
                DataLoader обучающей выборки.
            loss_fn : nn.Module
                Функция потерь.
            optimizer : torch.optim.Optimizer
                Оптимизатор модели.
            device : torch.device
                Устройство, на котором выполняются вычисления.

            Returns
            -------
            tuple[float, float]
                Средние значения loss и accuracy на обучающей выборке.
            \"\"\"
            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_count = 0

            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                # Прямой проход, расчет ошибки и шаг оптимизатора.
                optimizer.zero_grad()
                logits = model(images)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                batch_size = labels.size(0)
                train_loss_sum += loss.item() * batch_size
                train_correct += (logits.argmax(dim=1) == labels).sum().item()
                train_count += batch_size

            # Средние значения считаются по количеству изображений, а не по количеству batch.
            train_loss = train_loss_sum / train_count
            train_accuracy = train_correct / train_count

            return train_loss, train_accuracy
        """
    ),
    code(
        """
        def test_step(
            model: nn.Module,
            dataloader: DataLoader,
            loss_fn: nn.Module,
            device: torch.device,
        ) -> tuple[float, float]:
            \"\"\"
            Оценивает модель на validation или test выборке.

            Parameters
            ----------
            model : nn.Module
                Оцениваемая модель.
            dataloader : DataLoader
                DataLoader validation или test выборки.
            loss_fn : nn.Module
                Функция потерь.
            device : torch.device
                Устройство, на котором выполняются вычисления.

            Returns
            -------
            tuple[float, float]
                Средние значения loss и accuracy на выбранной выборке.
            \"\"\"
            model.eval()
            test_loss_sum = 0.0
            test_correct = 0
            test_count = 0

            with torch.inference_mode():
                for images, labels in dataloader:
                    images = images.to(device)
                    labels = labels.to(device)

                    # На оценке градиенты не считаются, поэтому веса модели не меняются.
                    logits = model(images)
                    loss = loss_fn(logits, labels)

                    batch_size = labels.size(0)
                    test_loss_sum += loss.item() * batch_size
                    test_correct += (logits.argmax(dim=1) == labels).sum().item()
                    test_count += batch_size

            test_loss = test_loss_sum / test_count
            test_accuracy = test_correct / test_count

            return test_loss, test_accuracy
        """
    ),
    code(
        """
        def train(
            model: nn.Module,
            train_dataloader: DataLoader,
            validation_dataloader: DataLoader,
            optimizer: torch.optim.Optimizer,
            loss_fn: nn.Module,
            epochs: int,
            device: torch.device,
        ) -> dict[str, list[float]]:
            \"\"\"
            Обучает модель и сохраняет историю изменения метрик.

            Parameters
            ----------
            model : nn.Module
                Обучаемая модель.
            train_dataloader : DataLoader
                DataLoader обучающей выборки.
            validation_dataloader : DataLoader
                DataLoader валидационной выборки.
            optimizer : torch.optim.Optimizer
                Оптимизатор модели.
            loss_fn : nn.Module
                Функция потерь.
            epochs : int
                Количество эпох обучения.
            device : torch.device
                Устройство, на котором выполняются вычисления.

            Returns
            -------
            dict[str, list[float]]
                История изменения loss и accuracy на train и validation.
            \"\"\"
            history = {
                "train_loss": [],
                "train_accuracy": [],
                "validation_loss": [],
                "validation_accuracy": [],
            }

            for epoch in range(1, epochs + 1):
                # Одна эпоха обучения и затем проверка на validation.
                train_loss, train_accuracy = train_step(
                    model=model,
                    dataloader=train_dataloader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    device=device,
                )
                validation_loss, validation_accuracy = test_step(
                    model=model,
                    dataloader=validation_dataloader,
                    loss_fn=loss_fn,
                    device=device,
                )

                history["train_loss"].append(train_loss)
                history["train_accuracy"].append(train_accuracy)
                history["validation_loss"].append(validation_loss)
                history["validation_accuracy"].append(validation_accuracy)

                print(
                    f"Epoch {epoch:02d}/{epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"train_accuracy={train_accuracy:.4f} | "
                    f"validation_loss={validation_loss:.4f} | "
                    f"validation_accuracy={validation_accuracy:.4f}"
                )

            return history
        """
    ),
    md(
        """
        ## Эксперимент 1: Данные без аугментации

        В первом эксперименте модель обучается на изображениях без случайных изменений. Данные только приводятся к одному размеру и переводятся в тензоры, поэтому этот эксперимент показывает базовое качество модели.
        """
    ),
    md(
        """
        ### Преобразование
        """
    ),
    code(
        """
        simple_transform = transforms.Compose([
            # Resize нужен, потому что DataLoader собирает изображения в batch одного размера.
            transforms.Resize(IMAGE_SIZE),
            # ToTensor переводит PIL-изображение в формат (C, H, W), с которым работает PyTorch.
            transforms.ToTensor(),
        ])
        """
    ),
    md(
        """
        ### Превью изображений
        """
    ),
    code(
        """
        fig, axes = plt.subplots(3, 4, figsize=(9, 6))
        axes = np.asarray(axes).reshape(-1)

        for axis, image_index in zip(axes, preview_indices):
            image = Image.open(image_paths[image_index]).convert("RGB")
            image_tensor = simple_transform(image)
            image_array = image_tensor.permute(1, 2, 0).numpy()

            label = targets[image_index]
            axis.imshow(image_array)
            axis.set_title(class_names[int(label)], fontsize=8)
            axis.axis("off")

        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ### Обучение
        """
    ),
    code(
        """
        simple_train_loader, simple_validation_loader, simple_test_loader = make_dataloaders(
            train_transform=simple_transform,
            eval_transform=simple_transform,
        )

        torch.manual_seed(SEED)
        simple_model = FurnitureCNN(
            input_shape=3,
            hidden_units=10,
            output_shape=len(class_names),
        ).to(device)
        simple_optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

        simple_history = train(
            model=simple_model,
            train_dataloader=simple_train_loader,
            validation_dataloader=simple_validation_loader,
            optimizer=simple_optimizer,
            loss_fn=loss_fn,
            epochs=EPOCHS,
            device=device,
        )
        """
    ),
    md(
        """
        ### Графики
        """
    ),
    code(
        """
        epochs = np.arange(1, len(simple_history["train_accuracy"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        axes[0].plot(epochs, simple_history["train_accuracy"], marker="o", label="train")
        axes[0].plot(epochs, simple_history["validation_accuracy"], marker="o", label="validation")
        axes[0].axhline(0.85, color="red", linestyle="--", label="порог 0.85")
        axes[0].set_title("Обычные данные: accuracy")
        axes[0].set_xlabel("Эпоха")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_ylim(0, 1)
        axes[0].legend()

        axes[1].plot(epochs, simple_history["train_loss"], marker="o", label="train")
        axes[1].plot(epochs, simple_history["validation_loss"], marker="o", label="validation")
        axes[1].set_title("Обычные данные: loss")
        axes[1].set_xlabel("Эпоха")
        axes[1].set_ylabel("Loss")
        axes[1].legend()

        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ### Оценка качества
        """
    ),
    code(
        """
        simple_model.eval()
        simple_test_loss_sum = 0.0
        simple_test_correct = 0
        simple_test_count = 0
        simple_true_labels = []
        simple_predicted_labels = []

        with torch.inference_mode():
            for images, labels in simple_test_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = simple_model(images)
                loss = loss_fn(logits, labels)
                predictions = logits.argmax(dim=1).cpu()

                # В этом же цикле считаем метрики и сохраняем предсказания для confusion matrix.
                batch_size = labels.size(0)
                simple_test_loss_sum += loss.item() * batch_size
                simple_test_correct += (predictions == labels.cpu()).sum().item()
                simple_test_count += batch_size

                simple_true_labels.append(labels.cpu())
                simple_predicted_labels.append(predictions)

        simple_test_loss = simple_test_loss_sum / simple_test_count
        simple_test_accuracy = simple_test_correct / simple_test_count
        simple_true_labels = torch.cat(simple_true_labels).numpy()
        simple_predicted_labels = torch.cat(simple_predicted_labels).numpy()

        print(f"Test loss: {simple_test_loss:.4f}")
        print(f"Test accuracy: {simple_test_accuracy:.4f}")
        """
    ),
    md(
        """
        ### Матрица ошибок
        """
    ),
    code(
        """
        simple_confusion_matrix = confusion_matrix(
            simple_true_labels,
            simple_predicted_labels,
            labels=np.arange(len(class_names)),
        )

        fig, axis = plt.subplots(figsize=(8, 7))
        display = ConfusionMatrixDisplay(
            confusion_matrix=simple_confusion_matrix,
            display_labels=class_names,
        )
        display.plot(ax=axis, cmap="Blues", values_format="d", colorbar=True)
        axis.set_title("Матрица ошибок: обычные данные")
        axis.set_xticklabels(class_names, rotation=45, ha="right")

        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Эксперимент 2: Данные с аугментацией

        Во втором эксперименте используется та же модель, но обучающие изображения дополнительно проходят аугментацию `TrivialAugmentWide`. Для каждого изображения случайно выбирается одно преобразование и его интенсивность, поэтому модель видит более разнообразные варианты исходных данных.
        """
    ),
    md(
        """
        ### Преобразование
        """
    ),
    code(
        """
        augmented_train_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            # TrivialAugmentWide случайно выбирает одну операцию и ее интенсивность.
            transforms.TrivialAugmentWide(num_magnitude_bins=31),
            transforms.ToTensor(),
        ])

        augmented_eval_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
        """
    ),
    md(
        """
        ### Превью изображений
        """
    ),
    code(
        """
        fig, axes = plt.subplots(3, 4, figsize=(9, 6))
        axes = np.asarray(axes).reshape(-1)

        for axis, image_index in zip(axes, preview_indices):
            image = Image.open(image_paths[image_index]).convert("RGB")
            image_tensor = augmented_train_transform(image)
            image_array = image_tensor.permute(1, 2, 0).numpy()

            label = targets[image_index]
            axis.imshow(image_array)
            axis.set_title(class_names[int(label)], fontsize=8)
            axis.axis("off")

        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ### Обучение
        """
    ),
    code(
        """
        augmented_train_loader, augmented_validation_loader, augmented_test_loader = make_dataloaders(
            train_transform=augmented_train_transform,
            eval_transform=augmented_eval_transform,
        )

        torch.manual_seed(SEED)
        augmented_model = FurnitureCNN(
            input_shape=3,
            hidden_units=10,
            output_shape=len(class_names),
        ).to(device)
        augmented_optimizer = torch.optim.Adam(augmented_model.parameters(), lr=1e-3)

        augmented_history = train(
            model=augmented_model,
            train_dataloader=augmented_train_loader,
            validation_dataloader=augmented_validation_loader,
            optimizer=augmented_optimizer,
            loss_fn=loss_fn,
            epochs=EPOCHS,
            device=device,
        )
        """
    ),
    md(
        """
        ### Графики
        """
    ),
    code(
        """
        epochs = np.arange(1, len(augmented_history["train_accuracy"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        axes[0].plot(epochs, augmented_history["train_accuracy"], marker="o", label="train")
        axes[0].plot(epochs, augmented_history["validation_accuracy"], marker="o", label="validation")
        axes[0].axhline(0.85, color="red", linestyle="--", label="порог 0.85")
        axes[0].set_title("TrivialAugmentWide: accuracy")
        axes[0].set_xlabel("Эпоха")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_ylim(0, 1)
        axes[0].legend()

        axes[1].plot(epochs, augmented_history["train_loss"], marker="o", label="train")
        axes[1].plot(epochs, augmented_history["validation_loss"], marker="o", label="validation")
        axes[1].set_title("TrivialAugmentWide: loss")
        axes[1].set_xlabel("Эпоха")
        axes[1].set_ylabel("Loss")
        axes[1].legend()

        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ### Оценка качества
        """
    ),
    code(
        """
        augmented_model.eval()
        augmented_test_loss_sum = 0.0
        augmented_test_correct = 0
        augmented_test_count = 0
        augmented_true_labels = []
        augmented_predicted_labels = []

        with torch.inference_mode():
            for images, labels in augmented_test_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = augmented_model(images)
                loss = loss_fn(logits, labels)
                predictions = logits.argmax(dim=1).cpu()

                # Повторный проход по test не нужен: здесь же собираются метрики и предсказания.
                batch_size = labels.size(0)
                augmented_test_loss_sum += loss.item() * batch_size
                augmented_test_correct += (predictions == labels.cpu()).sum().item()
                augmented_test_count += batch_size

                augmented_true_labels.append(labels.cpu())
                augmented_predicted_labels.append(predictions)

        augmented_test_loss = augmented_test_loss_sum / augmented_test_count
        augmented_test_accuracy = augmented_test_correct / augmented_test_count
        augmented_true_labels = torch.cat(augmented_true_labels).numpy()
        augmented_predicted_labels = torch.cat(augmented_predicted_labels).numpy()

        print(f"Test loss: {augmented_test_loss:.4f}")
        print(f"Test accuracy: {augmented_test_accuracy:.4f}")
        """
    ),
    md(
        """
        ### Матрица ошибок
        """
    ),
    code(
        """
        augmented_confusion_matrix = confusion_matrix(
            augmented_true_labels,
            augmented_predicted_labels,
            labels=np.arange(len(class_names)),
        )

        fig, axis = plt.subplots(figsize=(8, 7))
        display = ConfusionMatrixDisplay(
            confusion_matrix=augmented_confusion_matrix,
            display_labels=class_names,
        )
        display.plot(ax=axis, cmap="Blues", values_format="d", colorbar=True)
        axis.set_title("Матрица ошибок: TrivialAugmentWide")
        axis.set_xticklabels(class_names, rotation=45, ha="right")

        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Сравнение моделей
        """
    ),
    code(
        """
        model_names = ["Данные без аугментации", "Данные с аугментацией"]
        test_accuracies = [simple_test_accuracy, augmented_test_accuracy]
        test_losses = [simple_test_loss, augmented_test_loss]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].bar(model_names, test_accuracies, color=["#4c78a8", "#f58518"])
        axes[0].axhline(0.85, color="red", linestyle="--", label="порог 0.85")
        axes[0].set_title("Test accuracy")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_ylim(0, 1)
        axes[0].legend()

        axes[1].bar(model_names, test_losses, color=["#4c78a8", "#f58518"])
        axes[1].set_title("Test loss")
        axes[1].set_ylabel("Loss")

        plt.tight_layout()
        plt.show()
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
