from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbformat.notebooknode import NotebookNode


PRACTICE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PRACTICE_DIR / "СП-М-О-КЗ-2025-Практика-5-2-БабушкинМВ.ipynb"


def md(text: str) -> NotebookNode:
    """Создает markdown-ячейку notebook из многострочной строки."""
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str) -> NotebookNode:
    """Создает code-ячейку notebook из многострочной строки."""
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md("# Практическая работа 5.2. Обучение модели"),
    md("## Импорты"),
    code(
        """
        from __future__ import annotations

        import os
        from pathlib import Path

        PRACTICE_DIR = Path.cwd()
        CACHE_DIR = PRACTICE_DIR / ".cache"
        DATA_DIR = PRACTICE_DIR / ".data"
        MPL_CACHE_DIR = CACHE_DIR / "matplotlib"
        IPYTHON_CACHE_DIR = CACHE_DIR / "ipython"
        XDG_CACHE_DIR = CACHE_DIR / "xdg"
        MODEL_PATH = PRACTICE_DIR / "СП-М-О-КЗ-2025-Практика-5-2-БабушкинМВ.keras"

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
        import matplotlib.pyplot as plt
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
        from tensorflow.keras import layers
        from tensorflow.keras.models import Sequential

        set_height, set_width = 128, 128
        batch_size = 16
        epochs = 10

        PREPARED_DATASET_DIR = DATA_DIR / "flower-classification-7-classes-prepared"
        TRAIN_ROOT = PREPARED_DATASET_DIR / "train"
        VALIDATION_ROOT = PREPARED_DATASET_DIR / "validation"
        TEST_ROOT = PREPARED_DATASET_DIR / "test"

        plt.style.use("seaborn-v0_8-whitegrid")
        tf.keras.utils.set_random_seed(42)
        """
    ),
    md(
        """
        ## Описание данных

        В работе используется подготовленный flower-датасет, созданный в notebook `5.1`. Он содержит `7` классов:

        - `Daisy`
        - `Lavender`
        - `Lily`
        - `Orchid`
        - `Poppy`
        - `Rose`
        - `Sunflower`
        """
    ),
    md("## Подготовленный датасет"),
    code(
        """
        def load_prepared_dataset_info(
            train_root: Path,
            validation_root: Path,
            test_root: Path,
        ) -> tuple[list[str], dict[str, dict[str, int]]]:
            \"\"\"
            Считывает информацию о prepared-датасете.

            Parameters
            ----------
            train_root : Path
                Каталог train.
            validation_root : Path
                Каталог validation.
            test_root : Path
                Каталог test.

            Returns
            -------
            tuple[list[str], dict[str, dict[str, int]]]
                Список классов и количество изображений по split.
            \"\"\"
            if not all(root.is_dir() for root in (train_root, validation_root, test_root)):
                raise FileNotFoundError("Не найден prepared-датасет. Сначала выполните notebook 5.1.")

            class_names = sorted(directory.name for directory in train_root.iterdir() if directory.is_dir())
            split_counts: dict[str, dict[str, int]] = {}
            split_mapping = {
                "train": train_root,
                "validation": validation_root,
                "test": test_root,
            }

            for split_name, split_root in split_mapping.items():
                split_counts[split_name] = {}
                for class_name in class_names:
                    class_root = split_root / class_name
                    class_images = [path for path in class_root.iterdir() if path.is_file()]
                    split_counts[split_name][class_name] = len(class_images)

            return class_names, split_counts
        """
    ),
    code(
        """
        class_names, split_counts = load_prepared_dataset_info(
            TRAIN_ROOT,
            VALIDATION_ROOT,
            TEST_ROOT,
        )
        dataset_classes = len(class_names)

        print("Классы:", class_names)
        for split_name, counts in split_counts.items():
            print(f"{split_name.capitalize()}: {sum(counts.values())}")
            for class_name, count in counts.items():
                print(f"  {class_name}: {count}")
        """
    ),
    md("## Загрузка данных"),
    code(
        """
        def make_tf_dataset(
            split_root: Path,
            class_names: list[str],
            *,
            shuffle: bool,
        ) -> tf.data.Dataset:
            \"\"\"
            Создает TensorFlow Dataset из директории выборки.

            Parameters
            ----------
            split_root : Path
                Каталог выборки.
            class_names : list[str]
                Названия классов.
            shuffle : bool
                Нужно ли перемешивать изображения.

            Returns
            -------
            tf.data.Dataset
                Датасет изображений и меток.
            \"\"\"
            return tf.keras.preprocessing.image_dataset_from_directory(
                split_root,
                image_size=(set_height, set_width),
                batch_size=batch_size,
                class_names=class_names,
                shuffle=shuffle,
                seed=123,
            )
        """
    ),
    code(
        """
        training_images = make_tf_dataset(
            TRAIN_ROOT,
            class_names,
            shuffle=True,
        )

        validation_images = make_tf_dataset(
            VALIDATION_ROOT,
            class_names,
            shuffle=False,
        )

        test_images = make_tf_dataset(
            TEST_ROOT,
            class_names,
            shuffle=False,
        )
        """
    ),
    md("## Превью данных"),
    code(
        """
        image_batch, labels_batch = next(iter(training_images))

        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            plt.title(class_names[int(labels_batch[i])])
            plt.axis("off")
        plt.tight_layout()
        plt.show()
        """
    ),
    md("## Модель"),
    code(
        """
        model = Sequential([
          layers.Rescaling(1. / 255, input_shape=(set_height, set_width, 3)),
          layers.Conv2D(32, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(64, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(128, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Flatten(),
          layers.Dense(128, activation='relu'),
          layers.Dropout(0.40),
          layers.Dense(dataset_classes, activation='softmax')
        ])
        """
    ),
    code("model.summary()"),
    md("## Компиляция модели"),
    code(
        """
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        """
    ),
    md("## Обучение"),
    code(
        """
        CNN_model = model.fit(
            training_images,
            validation_data=validation_images,
            epochs=epochs
        )
        """
    ),
    md("## Графики обучения"),
    code(
        """
        accuracy = CNN_model.history['accuracy']
        val_accuracy = CNN_model.history['val_accuracy']

        loss = CNN_model.history['loss']
        val_loss = CNN_model.history['val_loss']

        epochs_range = range(epochs)
        """
    ),
    code(
        """
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, accuracy, label='Train Accuracy')
        plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Train Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.tight_layout()
        plt.show()
        """
    ),
    md("## Оценка на тестовой выборке"),
    code(
        """
        test_loss, test_accuracy = model.evaluate(test_images)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        """
    ),
    md("## Матрица ошибок"),
    code(
        """
        true_labels = []
        predicted_labels = []

        for image_batch, label_batch in test_images:
            predictions = model.predict(image_batch, verbose=0)
            predicted_batch = np.argmax(predictions, axis=1)
            true_labels.extend(label_batch.numpy())
            predicted_labels.extend(predicted_batch)

        cm = confusion_matrix(true_labels, predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
        ax.set_title("Матрица ошибок")
        plt.tight_layout()
        plt.show()
        """
    ),
    md("## Сохранение модели"),
    code(
        """
        model.save(MODEL_PATH)
        print(MODEL_PATH)
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
