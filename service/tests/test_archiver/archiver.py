# file: archiver.py
# directory: tests/test_archiver

import threading
import os
import sys
import shutil
import time
from queue import Queue
import config  # Импортируем модуль конфигурации
# Adjust the import paths if necessary
sys.path.append('../../')

from archiver import archiver_thread
from utils import configure_thread_logging, get_engine, get_session_factory, setup_database
from stats_collector import StatsCollector
from models import ArchivedImage
import json

def main():
    # Устанавливаем переменную окружения MACHINE_ID
    config.MACHINE_ID = int(os.environ.get('MACHINE_ID', '0'))

    # Директории для входных и выходных данных
    input_dir = 'input_data'
    output_dir = 'output_data'
    logs_dir = 'logs'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Проверяем наличие входных данных
    archive_info_file = os.path.join(input_dir, 'archive_info.json')
    if not os.path.exists(archive_info_file):
        print(f"Input file {archive_info_file} not found.")
        sys.exit(1)

    # Загружаем информацию для архивирования
    with open(archive_info_file, 'r') as f:
        archive_info = json.load(f)

    # Проверяем наличие директории с изображениями
    if not os.path.exists(archive_info['batch_dir']):
        print(f"Batch directory {archive_info['batch_dir']} not found.")
        sys.exit(1)

    # Очередь
    archive_queue = Queue()

    # Коллектор статистики
    stats_collector = StatsCollector()

    # Настройка логирования
    LOG_LEVEL = 'INFO'
    LOG_OUTPUT = 'file'
    log_filename = os.path.join(logs_dir, 'test_archiver.log')
    logger = configure_thread_logging('test_archiver', log_filename, LOG_LEVEL, LOG_OUTPUT)

    # Подготовка входных данных
    archive_queue.put(archive_info)
    archive_queue.put(None)  # Сигнал остановки

    # Настраиваем тестовую базу данных
    engine = get_engine()
    setup_database(engine)

    # Конфигурация архивации (используем локальную директорию для тестирования)
    archive_type = 'local'
    archive_config = {
        'archive_directory': os.path.join(output_dir, 'archive')
    }
    os.makedirs(archive_config['archive_directory'], exist_ok=True)

    # Модифицируем функции архиватора для обработки типа 'local'
    from archiver import archive_batch
    def archive_batch_local(batch_dir, batch_id, archive_type, archive_config, logger):
        archive_dir = archive_config['archive_directory']
        dest_dir = os.path.join(archive_dir, os.path.basename(batch_dir))
        shutil.copytree(batch_dir, dest_dir)
        archive_urls = {}
        for root, dirs, files in os.walk(dest_dir):
            for file in files:
                archive_urls[file] = os.path.join(dest_dir, file)
        return archive_urls

    # Заменяем функцию archive_batch
    import archiver
    archiver.archive_batch = archive_batch_local

    # Запускаем archiver_thread
    archiver_thread_instance = threading.Thread(target=archiver_thread, args=(
        archive_queue, engine, archive_type, archive_config, stats_collector, LOG_LEVEL, LOG_OUTPUT))
    archiver_thread_instance.start()

    # Ждем завершения потоков
    archive_queue.join()
    archiver_thread_instance.join()

    # Генерируем отчет
    report_file = os.path.join(output_dir, 'test_archiver_report.txt')
    with open(report_file, 'w') as f:
        f.write("Archiver Test Report\n")
        f.write("====================\n")
        f.write(f"Batches archived: {stats_collector.total_batches_archived}\n")
        f.write(f"Archive stored in {archive_config['archive_directory']}\n")
        f.write(f"Logs are saved in {logs_dir}\n")

    print(f"Test completed. Report saved to {report_file}")

if __name__ == "__main__":
    main()
