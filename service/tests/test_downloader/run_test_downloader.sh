#!/bin/bash

# file: tests/test_downloader/run_test_downloader.sh

# Проверяем наличие входных данных
if [ ! -f "input_data/test_page_info.json" ]; then
    echo "Input file input_data/test_page_info.json not found."
    exit 1
fi

# Резервируем предыдущие результаты
if [ -d "output_data" ]; then
    mv output_data "output_data_backup_$(date +%s)"
fi

# Создаем необходимые директории
mkdir -p output_data
mkdir -p logs

# Запускаем тест
python3 test_downloader.py

