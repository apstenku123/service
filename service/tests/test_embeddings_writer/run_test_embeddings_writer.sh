#!/bin/bash

# file: tests/test_embeddings_writer/run_test_embeddings_writer.sh

# Проверяем наличие входных данных
if [ ! -f "input_data/embeddings_data.json" ]; then
    echo "Input file input_data/embeddings_data.json not found."
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
cp ../test_processor/output_data/embeddings_data.json input_data/

# Запускаем тест
python3 test_embeddings_writer.py

