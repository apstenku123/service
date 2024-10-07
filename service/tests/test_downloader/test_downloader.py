# file: test_downloader.py
# directory: tests/test_downloader

import threading
import os
import sys
import shutil
import time
from queue import Queue

# Adjust the import paths if necessary
sys.path.append('../../')

from downloader import downloader_thread
from utils import configure_thread_logging, get_engine, get_session_factory, setup_database
from stats_collector import StatsCollector
from models import Image, Batch, BatchImage
import json
import config  # Импортируем модуль конфигурации

def main():
    # Устанавливаем переменную окружения MACHINE_ID
    config.MACHINE_ID = int(os.environ.get('MACHINE_ID', '0'))

    # Директории для входных и выходных данных
    input_dir = 'input_data'
    output_dir = 'output_data'
    logs_dir = 'logs'
    download_dir = os.path.join(output_dir, 'downloads')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)

    # Проверяем наличие входных данных
    test_page_info_file = os.path.join(input_dir, 'test_page_info.json')
    if not os.path.exists(test_page_info_file):
        print(f"Input file {test_page_info_file} not found.")
        sys.exit(1)

    # Загружаем информацию о тестовых страницах
    with open(test_page_info_file, 'r') as f:
        test_page_info = json.load(f)

    # Очереди
    page_queue = Queue()
    batch_queue = Queue()
    db_queue = Queue()
    batch_ready_queue = Queue()

    # Коллектор статистики
    stats_collector = StatsCollector()

    # Настройка логирования
    LOG_LEVEL = 'INFO'
    LOG_OUTPUT = 'file'
    log_filename = os.path.join(logs_dir, 'test_downloader.log')
    logger = configure_thread_logging('test_downloader', log_filename, LOG_LEVEL, LOG_OUTPUT)

    # Подготовка входных данных
    for page_info in test_page_info:
        page_queue.put(page_info)
    page_queue.put(None)  # Сигнал остановки

    # Настраиваем тестовую базу данных
    engine = get_engine()
    setup_database(engine)

    # Заглушки для batch_queue
    def dummy_batch_queue_consumer(batch_queue, output_dir):
        while True:
            batch_info = batch_queue.get()
            if batch_info is None:
                batch_queue.task_done()
                break
            # Сохраняем batch_info для следующего теста
            output_file = os.path.join(output_dir, 'batch_info.json')
            with open(output_file, 'w') as f:
                json.dump(batch_info, f)
            batch_queue.task_done()

    # Запускаем заглушку потребителя batch_queue
    batch_queue_consumer_thread = threading.Thread(target=dummy_batch_queue_consumer, args=(batch_queue, output_dir))
    batch_queue_consumer_thread.start()

    # Запускаем downloader_thread
    download_threads = 4  # Настройте по необходимости
    archive_enabled = False

    downloader = threading.Thread(target=downloader_thread, args=(
        page_queue, batch_queue, db_queue, batch_ready_queue, download_dir, download_threads, stats_collector, LOG_LEVEL, LOG_OUTPUT, archive_enabled))
    downloader.start()

    # Запускаем db_writer_thread
    from db_writer import db_writer_thread
    db_writer = threading.Thread(target=db_writer_thread, args=(
        db_queue, batch_ready_queue, engine, stats_collector, LOG_LEVEL, LOG_OUTPUT))
    db_writer.start()

    # Ждем завершения потоков
    page_queue.join()
    batch_queue.join()
    db_queue.join()
    batch_ready_queue.join()

    # Останавливаем потоки
    batch_queue.put(None)
    db_queue.put(None)

    downloader.join()
    db_writer.join()
    batch_queue_consumer_thread.join()

    # Генерируем отчет
    report_file = os.path.join(output_dir, 'test_downloader_report.txt')
    with open(report_file, 'w') as f:
        f.write("Downloader Test Report\n")
        f.write("====================\n")
        f.write(f"Files downloaded: {stats_collector.total_files_downloaded}\n")
        f.write(f"Batch info saved in {output_dir}/batch_info.json\n")
        f.write(f"Logs are saved in {logs_dir}\n")

    print(f"Test completed. Report saved to {report_file}")

if __name__ == "__main__":
    main()
