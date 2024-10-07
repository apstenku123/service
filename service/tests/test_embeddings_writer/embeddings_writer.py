# file: embeddings_writer.py
# directory: tests/test_embeddings_writer

import threading
import os
import sys
import shutil
import time
from queue import Queue

# Adjust the import paths if necessary
sys.path.append('../../')

from embeddings_writer import embeddings_writer_thread
from utils import configure_thread_logging, get_engine, get_session_factory, setup_database
from stats_collector import StatsCollector
from models import ImageEmbedding, Image
import json
import config  # Импортируем модуль конфигурации
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
    embeddings_file = os.path.join(input_dir, 'embeddings_data.json')
    if not os.path.exists(embeddings_file):
        print(f"Input file {embeddings_file} not found.")
        sys.exit(1)

    # Загружаем данные эмбеддингов
    with open(embeddings_file, 'r') as f:
        embeddings_data = json.load(f)

    # Очереди
    embeddings_queue = Queue()
    db_queue = Queue()

    # Коллектор статистики
    stats_collector = StatsCollector()

    # Настройка логирования
    LOG_LEVEL = 'INFO'
    LOG_OUTPUT = 'file'
    log_filename = os.path.join(logs_dir, 'test_embeddings_writer.log')
    logger = configure_thread_logging('test_embeddings_writer', log_filename, LOG_LEVEL, LOG_OUTPUT)

    # Подготовка входных данных
    embeddings_queue.put((1, embeddings_data))
    embeddings_queue.put(None)  # Сигнал остановки

    # Заглушка для db_queue
    def db_queue_consumer(db_queue):
        while True:
            db_task = db_queue.get()
            if db_task is None:
                db_queue.task_done()
                break
            db_task_type, data = db_task
            if db_task_type == 'mark_batch_processed':
                batch_id = data
                # Здесь можно имитировать обработку
                pass
            db_queue.task_done()

    db_consumer_thread = threading.Thread(target=db_queue_consumer, args=(db_queue,))
    db_consumer_thread.start()

    # Настраиваем тестовую базу данных
    engine = get_engine()
    setup_database(engine)

    # Запускаем embeddings_writer_thread
    embeddings_writer = threading.Thread(target=embeddings_writer_thread, args=(
        embeddings_queue, db_queue, engine, stats_collector, LOG_LEVEL, LOG_OUTPUT))
    embeddings_writer.start()

    # Ждем завершения потоков
    embeddings_queue.join()
    db_queue.join()

    # Останавливаем потоки
    db_queue.put(None)
    embeddings_writer.join()
    db_consumer_thread.join()

    # Генерируем отчет
    report_file = os.path.join(output_dir, 'test_embeddings_writer_report.txt')
    with open(report_file, 'w') as f:
        f.write("Embeddings Writer Test Report\n")
        f.write("=============================\n")
        f.write(f"Embeddings uploaded: {stats_collector.total_embeddings_uploaded}\n")
        f.write(f"Logs are saved in {logs_dir}\n")

    print(f"Test completed. Report saved to {report_file}")

if __name__ == "__main__":
    main()
