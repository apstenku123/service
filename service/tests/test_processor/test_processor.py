# file: tests/test_processor/test_processor.py

import threading
import os
import sys
import shutil
import time
from queue import Queue

# Adjust the import paths if necessary
sys.path.append('../../')

from processor import processing_thread
from utils import configure_thread_logging, get_engine, setup_database
from stats_collector import StatsCollector
from models import ImageEmbedding, Image, BatchImage
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import json

def main():
    # Директории для входных и выходных данных
    input_dir = 'input_data'
    output_dir = 'output_data'
    logs_dir = 'logs'
    report_dir = os.path.join(output_dir, 'reports')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # Проверяем наличие входных данных
    batch_info_file = os.path.join(input_dir, 'batch_info.json')
    if not os.path.exists(batch_info_file):
        print(f"Input file {batch_info_file} not found.")
        sys.exit(1)

    # Загружаем информацию о батче
    with open(batch_info_file, 'r') as f:
        batch_info = json.load(f)

    # Проверяем наличие скачанных изображений
    if not os.path.exists(batch_info['batch_dir']):
        print(f"Batch directory {batch_info['batch_dir']} not found.")
        sys.exit(1)

    # Очереди
    batch_queue = Queue()
    embeddings_queue = Queue()
    archive_queue = Queue()

    # Коллектор статистики
    stats_collector = StatsCollector()

    # Настройка логирования
    LOG_LEVEL = 'INFO'
    LOG_OUTPUT = 'file'
    log_filename = os.path.join(logs_dir, 'test_processor.log')
    logger = configure_thread_logging('test_processor', log_filename, LOG_LEVEL, LOG_OUTPUT)

    # Подготовка входных данных
    batch_queue.put(batch_info)
    batch_queue.put(None)  # Сигнал остановки

    # Заглушки для embeddings_queue и archive_queue
    def embeddings_consumer(embeddings_queue, output_dir):
        while True:
            embeddings_info = embeddings_queue.get()
            if embeddings_info is None:
                embeddings_queue.task_done()
                break
            batch_id, embeddings_data = embeddings_info
            output_file = os.path.join(output_dir, 'embeddings_data.json')
            with open(output_file, 'w') as f:
                json.dump(embeddings_data, f)
            embeddings_queue.task_done()

    def archive_queue_consumer(archive_queue):
        while True:
            archive_info = archive_queue.get()
            if archive_info is None:
                archive_queue.task_done()
                break
            # Сохраняем archive_info для следующего теста
            output_file = os.path.join(output_dir, 'archive_info.json')
            with open(output_file, 'w') as f:
                json.dump(archive_info, f)
            archive_queue.task_done()

    # Запускаем заглушки
    embeddings_consumer_thread = threading.Thread(target=embeddings_consumer, args=(embeddings_queue, output_dir))
    embeddings_consumer_thread.start()

    archive_consumer_thread = threading.Thread(target=archive_queue_consumer, args=(archive_queue,))
    archive_consumer_thread.start()

    # Инициализируем модели
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Открываем файл логов для изображений без лиц
    images_without_faces_log_file = open(os.path.join(logs_dir, 'images_without_faces.log'), 'a')

    # Настраиваем тестовую базу данных
    engine = get_engine()
    setup_database(engine)

    # Запускаем processing_thread
    batch_size = 8  # Настройте по необходимости
    processor = threading.Thread(target=processing_thread, args=(
        batch_queue, embeddings_queue, archive_queue, model, mtcnn, device, engine, batch_size, report_dir, stats_collector, LOG_LEVEL, LOG_OUTPUT, images_without_faces_log_file))
    processor.start()

    # Ждем завершения потоков
    batch_queue.join()
    embeddings_queue.join()
    archive_queue.join()

    # Останавливаем потоки
    embeddings_queue.put(None)
    archive_queue.put(None)

    processor.join()
    embeddings_consumer_thread.join()
    archive_consumer_thread.join()

    images_without_faces_log_file.close()

    # Генерируем отчет
    report_file = os.path.join(output_dir, 'test_processor_report.txt')
    with open(report_file, 'w') as f:
        f.write("Processor Test Report\n")
        f.write("=====================\n")
        f.write(f"Images processed: {stats_collector.total_images_processed}\n")
        f.write(f"Faces found: {stats_collector.total_faces_found}\n")
        f.write(f"Embeddings data saved in {output_dir}/embeddings_data.json\n")
        f.write(f"Archive info saved in {output_dir}/archive_info.json\n")
        f.write(f"Logs are saved in {logs_dir}\n")

    print(f"Test completed. Report saved to {report_file}")

if __name__ == "__main__":
    main()
