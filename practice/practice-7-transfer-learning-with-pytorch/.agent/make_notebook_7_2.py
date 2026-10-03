from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PRACTICE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_NOTEBOOK_PATH = PRACTICE_DIR / "Практическая работа 7.ipynb"
NOTEBOOK_PATH = PRACTICE_DIR / "СП-М-О-КЗ-2025-Практика-7-2-БабушкинМВ.ipynb"


DATA_LOADING_CELL = dedent(
    """
    # Увеличение количества данных и их нормализация
    # Нормализация для проверки
    data_transforms = {
         # Compose объединяет несколько преобразований в одну последовательность.
         'train': transforms.Compose([
             # RandomResizedCrop случайно вырезает фрагмент изображения
             # и приводит его к размеру 224x224.
             transforms.RandomResizedCrop(224),
             # RandomHorizontalFlip случайно отражает изображение по горизонтали.
             transforms.RandomHorizontalFlip(),
             # ToTensor переводит изображение из PIL-формата в тензор PyTorch.
             transforms.ToTensor(),
             # Normalize нормализует каналы по средним и стандартным отклонениям ImageNet.
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ]),
         # Для val аугментация не используется: только приведение размера и нормализация.
         'val': transforms.Compose([
            # Resize увеличивает меньшую сторону изображения до 256 пикселей.
            transforms.Resize(256),
             # CenterCrop берет центральный фрагмент размером 224x224.
             transforms.CenterCrop(224),
             # Перевод изображения в тензор.
             transforms.ToTensor(),
             # Нормализация по статистикам ImageNet.
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ]),
         # Для test используется та же схема, что и для валидации.
         'test': transforms.Compose([
            # Изменение размера изображения перед центральной обрезкой.
            transforms.Resize(256),
             # Центральная обрезка до 224x224.
             transforms.CenterCrop(224),
             # Перевод изображения в тензор.
             transforms.ToTensor(),
             # Нормализация по статистикам ImageNet.
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ]),
    }

    # prepared-датасет из notebook 7.1.
    data_dir = os.path.join(os.getcwd(), '.data', 'wonders-of-world-prepared')
    # ImageFolder считывает изображения из папок train / val / test.
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'test']}
    # DataLoader собирает изображения в батчи для обучения и оценки.
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val', 'test']}
    # Размеры выборок нужны для вычисления средней ошибки и accuracy.
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    # Список названий классов берется из train-папки.
    class_names = image_datasets['train'].classes

    # На Mac используем MPS, если он доступен.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    # Если MPS недоступен, проверяем CUDA.
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    # Иначе работаем на CPU.
    else:
        device = torch.device("cpu")
    """
).strip() + "\n"


MODEL_FT_CELL = dedent(
    """
    # Загружается ResNet18 с весами, предварительно обученными на ImageNet.
    model_ft = models.resnet18(pretrained=True)
    # in_features — количество входов в последний полносвязный слой.
    num_ftrs = model_ft.fc.in_features
    # Последний слой заменяется, чтобы число выходов совпадало с количеством классов.

    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    # Модель переносится на выбранное устройство.
    model_ft = model_ft.to(device)

    # CrossEntropyLoss используется для многоклассовой классификации.
    criterion = nn.CrossEntropyLoss()

    # Здесь обучаются все параметры модели, а не только последний слой.
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # StepLR уменьшает скорость обучения в 10 раз каждые 7 эпох.
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    """
).strip() + "\n"


MODEL_CONV_CELL = dedent(
    """
    # Загружается та же ResNet18 с предобученными весами.
    model_conv = torchvision.models.resnet18(pretrained=True)
    # Все параметры сверточной базы замораживаются.
    for param in model_conv.parameters():
        param.requires_grad = False

    # Определяется размер входа в последний полносвязный слой.
    num_ftrs = model_conv.fc.in_features
    # Последний слой заменяется новым классификатором под наш набор классов.
    model_conv.fc = nn.Linear(num_ftrs, len(class_names))

    # Модель переносится на выбранное устройство.
    model_conv = model_conv.to(device)

    # Функция потерь для многоклассовой классификации.
    criterion = nn.CrossEntropyLoss()

    # Оптимизируются только параметры последнего слоя.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # StepLR уменьшает скорость обучения в 10 раз каждые 7 эпох.
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    """
).strip() + "\n"


IMSHOW_CELL = dedent(
    """
    def imshow(inp, title=None):
        \"\"\"Imshow для тензора.\"\"\"
        # После make_grid тензор имеет формат (C, H, W), а matplotlib ожидает (H, W, C).
        inp = inp.numpy().transpose((1, 2, 0))
        # Это средние значения каналов, использованные при нормализации ImageNet.
        mean = np.array([0.485, 0.456, 0.406])
        # Это стандартные отклонения каналов, использованные при нормализации ImageNet.
        std = np.array([0.229, 0.224, 0.225])
        # Обратное преобразование нормализации, чтобы изображение можно было показать.
        inp = std * inp + mean
        # Ограничение значений диапазоном [0, 1] для корректного отображения.
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

    # Получение одного батча обучающего набора.
    inputs, classes = next(iter(dataloaders['train']))

    # make_grid объединяет несколько изображений из батча в одну общую сетку.
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])
    """
).strip() + "\n"


TEST_SECTION_CELLS = [
    nbf.v4.new_markdown_cell("### Оценка на тестовой выборке\n"),
    nbf.v4.new_code_cell(
        dedent(
            """
            from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

            def evaluate_model(model, criterion):
                # Сохраняется исходный режим модели, чтобы потом вернуть его обратно.
                was_training = model.training
                # Переключение в режим оценки отключает поведение, зависящее от обучения.
                model.eval()

                # Накопление суммы ошибки по всем батчам.
                running_loss = 0.0
                # Накопление количества правильных ответов.
                running_corrects = 0
                # Списки понадобятся для матрицы ошибок.
                true_labels = []
                predicted_labels = []

                # На этапе оценки градиенты не требуются.
                with torch.no_grad():
                    for inputs, labels in dataloaders['test']:
                        # Батч переносится на выбранное устройство.
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # Прямой проход модели.
                        outputs = model(inputs)
                        # По максимальному выходу выбирается предсказанный класс.
                        _, preds = torch.max(outputs, 1)
                        # Вычисляется ошибка на батче.
                        loss = criterion(outputs, labels)

                        # Накапливается сумма ошибки с учетом размера батча.
                        running_loss += loss.item() * inputs.size(0)
                        # Накапливается число правильных классификаций.
                        running_corrects += torch.sum(preds == labels.data)
                        # Истинные и предсказанные классы сохраняются для матрицы ошибок.
                        true_labels.extend(labels.cpu().numpy())
                        predicted_labels.extend(preds.cpu().numpy())

                # Средняя ошибка по всей тестовой выборке.
                test_loss = running_loss / dataset_sizes['test']
                # Accuracy считается как доля правильных ответов.
                test_acc = running_corrects.item() / dataset_sizes['test']

                print('Test Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))
                # Возврат исходного режима модели.
                model.train(mode=was_training)
                return true_labels, predicted_labels


            def plot_confusion_matrix(true_labels, predicted_labels, title):
                # По предсказаниям и истинным меткам строится матрица ошибок.
                cm = confusion_matrix(true_labels, predicted_labels)
                # Создается фигура для отображения матрицы.
                fig, ax = plt.subplots(figsize=(8, 8))
                # ConfusionMatrixDisplay рисует матрицу с подписями классов.
                display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
                display.plot(ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
                ax.set_title(title)
                plt.tight_layout()
                plt.show()
            """
        ).strip()
        + "\n"
    ),
    nbf.v4.new_code_cell(
        dedent(
            """
            true_labels, predicted_labels = evaluate_model(model_ft, criterion)
            plot_confusion_matrix(true_labels, predicted_labels, 'Матрица ошибок: model_ft')
            """
        ).strip()
        + "\n"
    ),
    nbf.v4.new_code_cell(
        dedent(
            """
            true_labels, predicted_labels = evaluate_model(model_conv, criterion)
            plot_confusion_matrix(true_labels, predicted_labels, 'Матрица ошибок: model_conv')
            """
        ).strip()
        + "\n"
    ),
]


def clear_outputs(notebook: nbf.NotebookNode) -> None:
    """Очищает выходы и счетчики выполнения в code-ячейках."""
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None


def main() -> None:
    """Создает notebook 7.2 на основе методички с заменой загрузки данных."""
    notebook = nbf.read(TEMPLATE_NOTEBOOK_PATH, as_version=4)

    notebook.cells[3].source = dedent(
        """
        Для загрузки данных используются пакеты torchvision и torch.utils.data.

        Данные берутся из подготовленного набора, сформированного в notebook 7.1.
        """
    ).strip() + "\n"
    notebook.cells[4].source = DATA_LOADING_CELL
    notebook.cells[6].source = IMSHOW_CELL
    notebook.cells[9].source = notebook.cells[9].source.replace(
        "epoch_acc = running_corrects.double() / dataset_sizes[phase]",
        "epoch_acc = running_corrects.item() / dataset_sizes[phase]",
    )
    notebook.cells[15].source = MODEL_FT_CELL
    notebook.cells[21].source = MODEL_CONV_CELL
    notebook.cells.extend(TEST_SECTION_CELLS)

    clear_outputs(notebook)
    nbf.write(notebook, NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
