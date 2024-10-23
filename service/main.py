# file: main.py
# directory: .

import os
import argparse
import threading
import logging
import time
import shutil

from queue import Queue
from utils import configure_thread_logging, get_engine, setup_database
from stats_collector import StatsCollector, stats_logger_thread
from downloader import downloader_thread, get_total_pages_for_query, process_page, download_image
from db_writer import db_writer_thread
from processor import processing_thread
from embeddings_writer import embeddings_writer_thread
from archiver import archiver_thread, archive_batch
import torch
from insightface.app import FaceAnalysis
import config  # Импортируем модуль конфигурации

condition = threading.Condition()

# Импортируем необходимые модели и функции
from models import (
    Checkpoint, BaseImageUrl, Image, Batch, BatchImage, ImageEmbedding, ArchivedImage, HostLog, BatchLog
)
from utils import get_session_factory, configure_thread_logging
from urllib.parse import urlparse
import cv2


def main():
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
    parser.add_argument('-ll','--log-level', type=str, default=os.environ.get('LOG_LEVEL', 'INFO'),
                        help='Logging level (default INFO)')
    parser.add_argument('-lo', '--log-output', type=str, choices=['file', 'console', 'both'],
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

    # Добавляем аргумент для выбора режима работы
    parser.add_argument('-m', '--mode', type=str, choices=['t', 's'], default='t',
                        help='Mode of operation: threaded (default) or sequential for debugging.')

    args = parser.parse_args()

    # Read environment variables
    config.MACHINE_ID = int(os.environ.get('MACHINE_ID', '0'))
    TOTAL_MACHINES = int(os.environ.get('TOTAL_MACHINES', '1'))
    DOWNLOAD_DIR = os.environ.get('DOWNLOAD_DIR', 'downloads')
    MAX_BATCHES_ON_DISK = int(os.environ.get('MAX_BATCHES_ON_DISK', '5'))
    config.current_batches_on_disk = 0  # Инициализируем переменную
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

    # Initialize models for extracting embeddings
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

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
    else:
        total_pages_to_process = args.limit

    mode = args.mode

    if mode == 't': # threaded
        # Task queues
        page_queue = Queue()
        batch_queue = Queue()
        embeddings_queue = Queue()
        db_queue = Queue()
        batch_ready_queue = Queue()
        archive_queue = Queue()

        # Event to stop the stats logger thread
        stop_event = threading.Event()

        # Fill the page queue for processing
        if args.query:
            for page_num in range(1, total_pages_to_process + 1):
                if (page_num % TOTAL_MACHINES) != config.MACHINE_ID:
                    continue  # This page is not for this machine

                page_info = {'page_number': page_num, 'query': search_query}
                page_queue.put(page_info)
        else:
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
        embeddings_writer = threading.Thread(target=embeddings_writer_thread, args=(embeddings_queue, db_queue, engine, stats_collector, LOG_LEVEL, LOG_OUTPUT))
        downloader = threading.Thread(target=downloader_thread, args=(
            page_queue, batch_queue, db_queue, batch_ready_queue, DOWNLOAD_DIR, DOWNLOAD_THREADS, stats_collector,
            LOG_LEVEL, LOG_OUTPUT, archive_enabled, condition, MAX_BATCHES_ON_DISK))

        processor = threading.Thread(target=processing_thread, args=(
            batch_queue, embeddings_queue, archive_queue, device, engine, BATCH_SIZE, REPORT_DIR, stats_collector,
            LOG_LEVEL, LOG_OUTPUT, images_without_faces_log_file, condition))

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
    else:
        # Sequential mode
        run_sequential(args, total_pages_to_process, DOWNLOAD_DIR, DOWNLOAD_THREADS, device, engine, BATCH_SIZE,
                       REPORT_DIR, stats_collector, LOG_LEVEL, LOG_OUTPUT, images_without_faces_log_file,
                       archive_enabled, archive_type, archive_config, condition, MAX_BATCHES_ON_DISK)

    logger.info("Application finished.")


def run_sequential(args, total_pages_to_process, DOWNLOAD_DIR, DOWNLOAD_THREADS, device, engine, BATCH_SIZE,
                   REPORT_DIR, stats_collector, LOG_LEVEL, LOG_OUTPUT, images_without_faces_log_file,
                   archive_enabled, archive_type, archive_config, condition, MAX_BATCHES_ON_DISK):
    logger = logging.getLogger('sequential')
    logger.info("Starting sequential processing mode.")

    # Создание сессии базы данных
    SessionFactory = get_session_factory(engine)
    session = SessionFactory()

    # Инициализация моделей
    app = FaceAnalysis(providers=['CUDAExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider'])
    app.prepare(ctx_id=0 if device.type == 'cuda' else -1)

    # Генерация списка страниц для обработки
    page_infos = generate_page_infos(args, total_pages_to_process)

    for page_info in page_infos:
        # 1. Обработка HTML страницы
        page_number = page_info['page_number']
        search_query = page_info.get('query')
        if search_query:
            page_url = f"http://camvideos.me/search/{search_query}?page={page_number}"
        else:
            page_url = f"http://camvideos.me/?page={page_number}"
        image_urls = process_page(page_url, stats_collector, LOG_LEVEL, LOG_OUTPUT)
        if not image_urls:
            logger.info(f"No images found on page {page_number}.")
            continue

        # 2. Сохранение батча в базу данных
        batch_info = store_batch_in_db(page_number, image_urls, session, LOG_LEVEL, LOG_OUTPUT)
        if not batch_info:
            logger.info(f"Batch for page {page_number} was not created.")
            continue

        # **Добавлено: Проверка лимита батчей на диске**
        with condition:
            while config.current_batches_on_disk >= MAX_BATCHES_ON_DISK:
                condition.wait()
            config.current_batches_on_disk += 1  # Увеличиваем счетчик

        try:
            # Inside the main loop in main()
            batch_info = download_batch_images(
                batch_info, DOWNLOAD_DIR, DOWNLOAD_THREADS, stats_collector, LOG_LEVEL, LOG_OUTPUT, archive_enabled,
                session
            )

            # Check if batch_info is None (no images to process)
            if not batch_info:
                logger.info(f"No images left in batch after removing failed downloads. Skipping batch.")
                continue  # Skip to the next page

            # Proceed with processing embeddings
            embeddings_data = process_embeddings(
                batch_info, app, device, stats_collector, LOG_LEVEL, LOG_OUTPUT, images_without_faces_log_file, session
            )

            # 5. Сохранение эмбеддингов в базу данных
            save_embeddings_to_db(batch_info['batch_id'], embeddings_data, session, stats_collector, LOG_LEVEL, LOG_OUTPUT)

            # 6. Архивация изображений (если включена)
            if archive_enabled:
                archive_batch_images(batch_info, archive_type, archive_config, session, stats_collector, LOG_LEVEL, LOG_OUTPUT)

            # 7. Обновление статистики и очистка
            logger.info(f"Completed processing for batch {batch_info['batch_id']}.")

            # Выводим статистику
            log_stats(stats_collector)
        finally:
            # **Добавлено: Уменьшаем счетчик и уведомляем**
            with condition:
                config.current_batches_on_disk -= 1
                condition.notify()

    session.close()
    logger.info("Sequential processing completed.")


def generate_page_infos(args, total_pages_to_process):
    page_infos = []
    if args.query:
        search_query = args.query
        total_pages = total_pages_to_process
        for page_num in range(1, total_pages + 1):
            if (page_num % int(os.environ.get('TOTAL_MACHINES', '1'))) != config.MACHINE_ID:
                continue
            page_info = {'page_number': page_num, 'query': search_query}
            page_infos.append(page_info)
    else:
        for page_num in range(args.start, args.start + args.limit):
            if (page_num % int(os.environ.get('TOTAL_MACHINES', '1'))) != config.MACHINE_ID:
                continue
            page_info = {'page_number': page_num, 'query': None}
            page_infos.append(page_info)
    return page_infos


def store_batch_in_db(page_number, image_urls, session, log_level, log_output):
    db_writer_logger = configure_thread_logging('db_writer_sequential', 'logs/db_writer_sequential.log', log_level, log_output)
    try:
        # Проверка существования чекпоинта
        page_url = f"http://camvideos.me/?page={page_number}"
        existing_checkpoint = session.query(Checkpoint).filter_by(page_url=page_url).first()
        if existing_checkpoint:
            db_writer_logger.info(f"Page {page_url} already processed. Skipping.")
            return None
        new_checkpoint = Checkpoint(page_url=page_url)
        session.add(new_checkpoint)
        session.commit()

        # Создание base_url
        base_urls = list(set([f"{urlparse(url).scheme}://{urlparse(url).netloc}" for url in image_urls]))
        base_url_str = base_urls[0]
        base_url = BaseImageUrl(base_url=base_url_str)
        session.add(base_url)
        session.commit()

        # Создание батча
        batch = Batch(page_number=page_number)
        session.add(batch)
        session.commit()

        # Добавление изображений
        images_data = []
        image_filenames = []
        image_paths = []  # Новое
        image_urls_list = []
        for img_url in image_urls:
            parsed_url = urlparse(img_url)
            filename = os.path.basename(parsed_url.path)
            path = os.path.dirname(parsed_url.path).lstrip('/')  # Удаляем ведущий '/'
            image = Image(base_url_id=base_url.id, path=path, filename=filename)
            images_data.append(image)
            image_filenames.append(filename)
            image_paths.append(path)
            image_urls_list.append(img_url)

        if not images_data:
            db_writer_logger.info(f"No images to insert for page {page_url}.")
            return None

        session.add_all(images_data)
        session.commit()

        # Создание batch_images
        batch_images_data = [BatchImage(batch_id=batch.id, image_id=img.id) for img in images_data]
        session.bulk_save_objects(batch_images_data)
        session.commit()

        batch_info = {
            'batch_id': batch.id,
            'batch_dir': f"batch_{batch.id}",
            'image_ids': [img.id for img in images_data],
            'filenames': image_filenames,
            'paths': image_paths,  # Добавлено
            'image_urls': image_urls_list,
        }
        db_writer_logger.info(f"Batch {batch.id} is ready.")
        return batch_info
    except Exception as e:
        session.rollback()
        db_writer_logger.error(f"Error in store_batch_in_db: {e}", exc_info=True)
        return None


def download_batch_images(batch_info, download_dir, download_threads, stats_collector, log_level, log_output, archive_enabled, session):
    downloader_logger = configure_thread_logging('downloader_sequential', 'logs/downloader_sequential.log', log_level, log_output)
    batch_id = batch_info['batch_id']
    batch_dir_name = batch_info['batch_dir']
    batch_dir = os.path.join(download_dir, batch_dir_name)
    os.makedirs(batch_dir, exist_ok=True)

    image_ids = batch_info['image_ids']
    filenames = batch_info['filenames']
    image_urls = batch_info['image_urls']

    id_to_filename = dict(zip(image_ids, filenames))
    id_to_url = dict(zip(image_ids, image_urls))

    images_to_download = []
    failed_downloads = []

    for img_id in image_ids:
        archived_image = session.query(ArchivedImage).filter_by(image_id=img_id).first()
        if archived_image and archive_enabled:
            archive_url = archived_image.archive_url
            local_path = os.path.join(batch_dir, id_to_filename[img_id])
            success = download_image(archive_url, local_path, downloader_logger, stats_collector)
            if not success:
                images_to_download.append(img_id)
        else:
            images_to_download.append(img_id)

    if images_to_download:
        for img_id in images_to_download:
            filename = id_to_filename[img_id]
            img_url = id_to_url[img_id]
            local_path = os.path.join(batch_dir, filename)
            success = download_image(img_url, local_path, downloader_logger, stats_collector)
            if not success:
                downloader_logger.error(f"Failed to download image {img_url}")
                failed_downloads.append(img_id)

    # Handle failed downloads
    if failed_downloads:
        downloader_logger.warning(f"Failed to download images: {[id_to_url[id] for id in failed_downloads]}")
        # Remove image records from database for failed downloads
        try:
            # Remove from BatchImage
            session.query(BatchImage).filter(
                BatchImage.batch_id == batch_id,
                BatchImage.image_id.in_(failed_downloads)
            ).delete(synchronize_session=False)
            # Remove from Image
            session.query(Image).filter(Image.id.in_(failed_downloads)).delete(synchronize_session=False)
            session.commit()
            downloader_logger.info(f"Removed failed images from database.")
        except Exception as e:
            session.rollback()
            downloader_logger.error(f"Error removing failed images from database: {e}", exc_info=True)

    # Update batch_info to exclude failed downloads
    remaining_image_ids = [img_id for img_id in image_ids if img_id not in failed_downloads]
    if not remaining_image_ids:
        downloader_logger.warning(f"No images left in batch {batch_id} after removing failed downloads.")
        # Remove batch from database
        try:
            session.query(BatchImage).filter(BatchImage.batch_id == batch_id).delete(synchronize_session=False)
            session.query(Batch).filter(Batch.id == batch_id).delete(synchronize_session=False)
            session.commit()
            downloader_logger.info(f"Removed batch {batch_id} from database.")
        except Exception as e:
            session.rollback()
            downloader_logger.error(f"Error removing batch {batch_id} from database: {e}", exc_info=True)
        # Delete batch directory
        shutil.rmtree(batch_dir, ignore_errors=True)
        return None  # Indicate that the batch has no images to process
    else:
        # Update batch_info
        batch_info['image_ids'] = remaining_image_ids
        batch_info['filenames'] = [id_to_filename[img_id] for img_id in remaining_image_ids]
        batch_info['image_urls'] = [id_to_url[img_id] for img_id in remaining_image_ids]
        batch_info['paths'] = [batch_info['paths'][image_ids.index(img_id)] for img_id in remaining_image_ids]

    batch_info['batch_dir'] = batch_dir
    return batch_info


def process_embeddings(batch_info, app, device, stats_collector, log_level, log_output, images_without_faces_log_file, session):
    embedding_processor_logger = configure_thread_logging('embedding_processor_sequential', 'logs/embedding_processor_sequential.log', log_level, log_output)
    batch_id = batch_info['batch_id']
    batch_dir = batch_info['batch_dir']
    image_ids = batch_info['image_ids']
    filenames = batch_info['filenames']
    image_urls = batch_info['image_urls']
    filename_to_id = dict(zip(filenames, image_ids))
    filename_to_url = dict(zip(filenames, image_urls))

    embeddings_data = []
    try:
        image_paths = [os.path.join(batch_dir, filename) for filename in filenames]
        valid_image_ids = []
        valid_filenames = []
        valid_image_urls = []
        images_data = []

        for idx, path in enumerate(image_paths):
            img = cv2.imread(path)
            if img is not None:
                images_data.append(img)
                valid_image_ids.append(filename_to_id[filenames[idx]])
                valid_filenames.append(filenames[idx])
                valid_image_urls.append(filename_to_url[filenames[idx]])
            else:
                embedding_processor_logger.error(f"Failed to load image: {path}")

        total_images = len(images_data)
        total_faces = 0
        images_with_faces = 0
        images_without_faces = 0

        for idx_in_batch, img in enumerate(images_data):
            faces = app.get(img)
            num_faces = len(faces)
            total_faces += num_faces
            if num_faces > 0:
                images_with_faces += 1
                stats_collector.increment_faces_found(num_faces)
                stats_collector.increment_images_with_faces()
                for face in faces:
                    embedding = face.embedding.flatten().tolist()
                    embeddings_data.append({
                        'image_id': valid_image_ids[idx_in_batch],
                        'filename': valid_filenames[idx_in_batch],
                        'embedding': embedding
                    })
            else:
                images_without_faces += 1
                stats_collector.increment_images_without_faces()
                embedding_processor_logger.info(f"No faces detected in image: {valid_filenames[idx_in_batch]}")
                images_without_faces_log_file.write(f"{valid_image_urls[idx_in_batch]}\n")

        # Обновление статистики
        stats_collector.increment_images_processed(len(valid_image_ids))
        stats_collector.increment_batches_processed()

        # Mark images as processed
        session.query(Image).filter(Image.id.in_(valid_image_ids)).update({"processed": True}, synchronize_session=False)

        # Mark batch as processed
        batch = session.query(Batch).filter_by(id=batch_id).first()
        batch.processed = True
        session.commit()

        # Удаление временной директории
        shutil.rmtree(batch_dir)
        embedding_processor_logger.info(f"Removed temporary directory for batch {batch_id}")

        return embeddings_data
    except Exception as e:
        session.rollback()
        embedding_processor_logger.error(f"Error processing embeddings for batch {batch_id}: {e}", exc_info=True)
        return []


def save_embeddings_to_db(batch_id, embeddings_data, session, stats_collector, log_level, log_output):
    embeddings_writer_logger = configure_thread_logging('embeddings_writer_sequential', 'logs/embeddings_writer_sequential.log', log_level, log_output)
    try:
        if not embeddings_data:
            embeddings_writer_logger.info(f"No embeddings to save for batch {batch_id}")
            return

        embeddings_objects = []
        for data in embeddings_data:
            embedding_vector = data['embedding']
            if len(embedding_vector) != 512:
                embeddings_writer_logger.error(f"Embedding length is not 512 for image_id {data['image_id']}")
                continue

            embedding = ImageEmbedding(
                image_id=data['image_id'],
                filename=data['filename'],
                insightface_embedding=embedding_vector
            )
            embeddings_objects.append(embedding)
        session.bulk_save_objects(embeddings_objects)
        session.commit()
        embeddings_writer_logger.info(f"Embeddings for batch {batch_id} committed to database.")
        stats_collector.increment_embeddings_uploaded(len(embeddings_objects))
    except Exception as e:
        session.rollback()
        embeddings_writer_logger.error(f"Error saving embeddings for batch {batch_id}: {e}", exc_info=True)


def archive_batch_images(batch_info, archive_type, archive_config, session, stats_collector, log_level, log_output):
    archiver_logger = configure_thread_logging('archiver_sequential', 'logs/archiver_sequential.log', log_level, log_output)
    batch_id = batch_info['batch_id']
    batch_dir = batch_info['batch_dir']
    filenames = batch_info['filenames']
    image_ids = batch_info['image_ids']
    filename_to_id = dict(zip(filenames, image_ids))

    archiver_logger.info(f"Starting archiving for batch {batch_id}")

    # Архивация изображений
    archive_urls = archive_batch(batch_dir, batch_id, archive_type, archive_config, archiver_logger)

    # Сохранение archive_urls в базе данных
    try:
        for filename, archive_url in archive_urls.items():
            img_id = filename_to_id.get(filename)
            if img_id:
                archived_image = ArchivedImage(image_id=img_id, archive_url=archive_url)
                session.add(archived_image)
        session.commit()
        archiver_logger.info(f"Archived images for batch {batch_id} stored in database.")
    except Exception as e:
        session.rollback()
        archiver_logger.error(f"Error storing archive URLs for batch {batch_id}: {e}")

    stats_collector.increment_batches_archived()


def log_stats(stats_collector):
    stats = stats_collector.reset()
    # Формируем сообщение статистики
    message = (
        f"Total Images Processed: {stats['total_images_processed']}\n"
        f"Total Faces Found: {stats['total_faces_found']}\n"
        f"Images with Faces: {stats['images_with_faces']}\n"
        f"Images without Faces: {stats['images_without_faces']}\n"
        f"Total Embeddings Uploaded: {stats['total_embeddings_uploaded']}\n"
        f"Total Batches Archived: {stats['total_batches_archived']}\n"
    )
    print(message)


if __name__ == "__main__":
    main()
