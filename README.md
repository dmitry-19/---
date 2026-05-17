Проект/
├── main.py                  # CLI для пакетной обработки троек (AB1 + FASTA)
├── train.py                 # Обучение нейросети-корректора
├── inference.py             # Интерактивное тестирование обученной модели
├── basecaller.pt            # Сохранённые веса лучшей модели
├── pipeline.log / training.log / inference.log
├── data/                    # Входные файлы (AB1 и FASTA)
├── output/                  # Визуализации и FASTA-результаты инференса
├── datasets/                # Сгенерированные обучающие выборки (X.npy, y.npy, meta.npy)
└── src/                     # Исходные модули
    ├── logger.py            # Настройка логирования
    ├── reader.py            # Чтение и предобработка AB1-файлов
    ├── aligner.py           # Парное выравнивание консенсуса с ридами
    ├── hybrid_aligner.py    # Гибридное выравнивание ридов (буквы + DTW)
    ├── visualizer.py        # Графики (хроматограммы, DTW-пути, трёхстороннее выравнивание)
    ├── file_utils.py        # Поиск и группировка файлов в data/
    ├── dataset_builder.py   # Создание датасета из карты сопоставлений
    ├── model.py             # Архитектура нейросети BaseCallerNet
    └── dataset_loader.py    # PyTorch Dataset для загрузки .npy-файлов
