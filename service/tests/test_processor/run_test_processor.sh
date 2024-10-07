#!/bin/bash

# file: tests/test_processor/run_test_processor.sh

# Проверяем наличие входных данных
if [ ! -f "input_data/batch_info.json" ]; then
    echo "Input file input_data/batch_info.json not found."
    exit 1
fi

# Резервируем предыдущие результаты
if [ -d "output_data" ]; then
    mv output_data "output_data_backup_$(date +%s)"
fi

# Создаем необходимые директории
mkdir -p output_data
mkdir -p logs

# Копируем необходимые файлы из предыдущего теста
cp ../test_downloader/output_data/batch_info.json input_data/

# Запускаем тест
python3 test_processor.py

