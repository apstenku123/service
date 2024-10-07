# file: main.py
# directory: .
import os
import argparse
import threading
import logging
import time
from queue import Queue
from utils import configure_thread_logging, get_engine, setup_database
from stats_collector import StatsCollector, stats_logger_thread
from downloader import downloader_thread, get_total_pages_for_query
from db_writer import db_writer_thread
from processor import processing_thread
from embeddings_writer import embeddings_writer_thread
from archiver import archiver_thread
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import config  # Импортируем модуль конфигурации

def main():
    # global MACHINE_ID
    parser = argparse.ArgumentParser(description='Script for parsing, downloading, and processing images.')
    parser.add_argument('-l', '--limit', type=int, default=1, help='Number of pages to process')
    parser.add_argument('-s', '--start', type=int, default=1, help='Starting page number')
    parser.add_argument('-dt', '--download-threads', type=int, default=int(os.environ.get('DOWNLOAD_THREADS', 8)),
                        help='Number of threads for downloading images in a batch (default 8)')
    parser.add_argument('-bs', '--batch-size', type=int, default=int(os.environ.get('BATCH_SIZE', 16)),
                        help='Batch size for processing images (default 16)')
    parser.add_argument('-rd', '--report-dir', type=str, default=os.environ.get('REPORT_DIR', 'reports'),
                        help='Directory for saving reports (default "reports")')
    parser.add_argument('-si', '--stats-interval', type=int, default=int(os.environ.get('STATS_INTERVAL', 10)),
                        help='Interval in seconds for statistics logging (default 10)')
    parser.add_argument('--log-level', type=str, default=os.environ.get('LOG_LEVEL', 'INFO'),
                        help='Logging level (default INFO)')
    parser.add_argument('--log-output', type=str, choices=['file', 'console', 'both'],
                        default=os.environ.get('LOG_OUTPUT', 'file'),
                        help='Logging output: file, console, or both (default file)')
    parser.add_argument('--loggers', type=str, default=os.environ.get('LOGGERS', ''),
                        help='Comma-separated list of loggers to include (default all)')
    parser.add_argument('--archive', action='store_true', help='Enable archiving of images after processing (default False)')
    parser.add_argument('--archive-type', type=str, choices=['s3', 'azure', 'ftp', 'sftp'], help='Type of archive storage')
    parser.add_argument('--archive-config', type=str, help='Path to archive configuration file')
    parser.add_argument('--archive-threads', type=int, default=int(os.environ.get('ARCHIVE_THREADS', 4)),
                        help='Number of archiver threads (default 4)')
    parser.add_argument('--service', action='store_true', help='Run as a web service')
    parser.add_argument('--port', type=int, default=8070, help='Port for the web service (default 8070)')

    # Добавляем новый аргумент командной строки для поисковой строки
    parser.add_argument('-q', '--query', type=str, help='Search query string')

    args = parser.parse_args()

    # Read environment variables
    config.MACHINE_ID = int(os.environ.get('MACHINE_ID', '0'))
    TOTAL_MACHINES = int(os.environ.get('TOTAL_MACHINES', '1'))
    DOWNLOAD_DIR = os.environ.get('DOWNLOAD_DIR', 'downloads')
    MAX_BATCHES_ON_DISK = int(os.environ.get('MAX_BATCHES_ON_DISK', '5'))
    DOWNLOAD_THREADS = args.download_threads
    BATCH_SIZE = args.batch_size
    REPORT_DIR = args.report_dir
    STATS_INTERVAL = args.stats_interval
    LOG_LEVEL = getattr(logging, args.log_level.upper(), logging.INFO)
    LOG_OUTPUT = args.log_output
    LOGGERS = args.loggers.split(',') if args.loggers else []

    # Archiving settings
    archive_enabled = args.archive or os.environ.get('ARCHIVE', 'False') == 'True'
    archive_type = args.archive_type or os.environ.get('ARCHIVE_TYPE')
    archive_config_path = args.archive_config or os.environ.get('ARCHIVE_CONFIG')
    ARCHIVE_THREADS = args.archive_threads

    if archive_enabled and not archive_type:
        print("Archiving is enabled but no archive type is specified.")
        return

    if archive_enabled and not archive_config_path:
        print("Archiving is enabled but no archive configuration file is provided.")
        return

    if archive_enabled:
        # Load archive configuration
        import json
        with open(archive_config_path, 'r') as f:
            archive_config = json.load(f)
    else:
        archive_config = {}

    print(f"MACHINE_ID: {config.MACHINE_ID}, TOTAL_MACHINES: {TOTAL_MACHINES}, DOWNLOAD_DIR: {DOWNLOAD_DIR}, "
          f"MAX_BATCHES_ON_DISK: {MAX_BATCHES_ON_DISK}, DOWNLOAD_THREADS: {DOWNLOAD_THREADS}, "
          f"BATCH_SIZE: {BATCH_SIZE}, REPORT_DIR: {REPORT_DIR}, STATS_INTERVAL: {STATS_INTERVAL}, "
          f"LOG_LEVEL: {args.log_level}, LOG_OUTPUT: {LOG_OUTPUT}, LOGGERS: {LOGGERS}, "
          f"ARCHIVE_ENABLED: {archive_enabled}, ARCHIVE_TYPE: {archive_type}, ARCHIVE_THREADS: {ARCHIVE_THREADS}")

    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    # Set up logging for main
    log_filename = os.path.join('logs', 'main', f'main_{config.MACHINE_ID}.log')
    logger = configure_thread_logging('main', log_filename, LOG_LEVEL, LOG_OUTPUT)
    logger.info("Application started.")

    # Set up the database
    engine = get_engine()
    setup_database(engine)

    # Initialize the statistics collector
    stats_collector = StatsCollector()

    # Task queues
    page_queue = Queue()
    batch_queue = Queue(maxsize=MAX_BATCHES_ON_DISK)
    embeddings_queue = Queue()
    db_queue = Queue()
    batch_ready_queue = Queue()
    archive_queue = Queue()

    # Initialize models for extracting embeddings
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    mtcnn = MTCNN(keep_all=True, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Event to stop the stats logger thread
    stop_event = threading.Event()

    # Open log file for images without faces
    log_file_path = os.path.join('logs', f'images_without_faces_{config.MACHINE_ID}.log')
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    images_without_faces_log_file = open(log_file_path, 'a')

    # Вычисляем total_pages_to_process перед запуском потоков
    total_pages_to_process = 0  # Инициализируем переменную

    # Если задан поисковый запрос, обрабатываем его
    if args.query:
        search_query = args.query
        total_pages = get_total_pages_for_query(search_query)
        total_pages_to_process = total_pages
        logger.info(f"Total pages for query '{search_query}': {total_pages}")

        for page_num in range(1, total_pages + 1):
            if (page_num % TOTAL_MACHINES) != config.MACHINE_ID:
                continue  # This page is not for this machine

            page_info = {'page_number': page_num, 'query': search_query}
            page_queue.put(page_info)
    else:
        # Fill the page queue for processing
        total_pages_to_process = args.limit
        for page_num in range(args.start, args.start + args.limit):
            if (page_num % TOTAL_MACHINES) != config.MACHINE_ID:
                continue  # This page is not for this machine

            page_info = {'page_number': page_num, 'query': None}
            page_queue.put(page_info)

    # Start threads
    stats_logger = threading.Thread(target=stats_logger_thread, args=(
        stats_collector,
        STATS_INTERVAL,
        stop_event,
        LOG_LEVEL,
        LOG_OUTPUT,
        total_pages_to_process,  # Передаем total_pages_to_process
        page_queue,
        batch_queue,
        embeddings_queue,
        db_queue
    ))
    db_writer = threading.Thread(target=db_writer_thread, args=(db_queue, batch_ready_queue, engine, stats_collector, LOG_LEVEL, LOG_OUTPUT))
    downloader = threading.Thread(target=downloader_thread, args=(page_queue, batch_queue, db_queue, batch_ready_queue, DOWNLOAD_DIR, DOWNLOAD_THREADS, stats_collector, LOG_LEVEL, LOG_OUTPUT, archive_enabled))
    embeddings_writer = threading.Thread(target=embeddings_writer_thread, args=(embeddings_queue, db_queue, engine, stats_collector, LOG_LEVEL, LOG_OUTPUT))
    processor = threading.Thread(target=processing_thread, args=(batch_queue, embeddings_queue, archive_queue, model, mtcnn, device, engine, BATCH_SIZE, REPORT_DIR, stats_collector, LOG_LEVEL, LOG_OUTPUT, images_without_faces_log_file))

    archiver_threads = []
    if archive_enabled:
        for _ in range(ARCHIVE_THREADS):
            archiver = threading.Thread(target=archiver_thread, args=(archive_queue, engine, archive_type, archive_config, stats_collector, LOG_LEVEL, LOG_OUTPUT))
            archiver_threads.append(archiver)
            archiver.start()

    db_writer.start()
    downloader.start()
    embeddings_writer.start()
    processor.start()
    stats_logger.start()

    logger.info("All threads started.")

    # Завершаем очередь страниц
    page_queue.put(None)
    page_queue.join()

    # Wait until all batches are processed
    batch_queue.join()
    embeddings_queue.join()
    db_queue.join()

    # Terminate remaining threads
    batch_queue.put(None)
    embeddings_queue.put(None)
    db_queue.put(None)
    batch_ready_queue.put(None)

    downloader.join()
    db_writer.join()
    embeddings_writer.join()
    processor.join()

    # Close archive queue and wait for archiver threads
    if archive_enabled:
        for _ in archiver_threads:
            archive_queue.put(None)
        archive_queue.join()
        for archiver in archiver_threads:
            archiver.join()

    # Stop the stats logger thread
    stop_event.set()
    stats_logger.join()

    # Close the images_without_faces log file
    images_without_faces_log_file.close()
    logger.info("All batches processed.")

if __name__ == "__main__":
    main()
