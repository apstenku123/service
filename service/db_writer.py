# file: db_writer.py
# directory: .

import os
import threading
from urllib.parse import urlparse

from utils import get_session_factory, configure_thread_logging
from models import HostLog, Checkpoint, BaseImageUrl, Image, Batch, BatchImage, ArchivedImage
import socket
import traceback
import config  # Импортируем модуль конфигурации

def db_writer_thread(db_queue, batch_ready_queue, engine, stats_collector, log_level, log_output):
    # Set up logger
    log_filename = f'logs/db_writer/db_writer_{config.MACHINE_ID}.log'
    db_writer_logger = configure_thread_logging('db_writer', log_filename, log_level, log_output)

    SessionFactory = get_session_factory(engine)
    session = SessionFactory()
    host_name = socket.gethostname()

    # Проверка существования HostLog
    existing_host_log = session.query(HostLog).filter_by(host_name=host_name, function_name='db_writer', log_file=log_filename).first()
    if not existing_host_log:
        host_log = HostLog(host_name=host_name, function_name='db_writer', log_file=log_filename)
        session.add(host_log)
        session.commit()
    else:
        host_log = existing_host_log

    while True:
        db_task = db_queue.get()
        if db_task is None:
            db_queue.task_done()
            break  # Termination signal

        task_type, data = db_task

        try:
            if task_type == 'store_batch':
                page_number, image_urls = data
                # Обработка чекпоинта
                page_url = f"http://camvideos.me/?page={page_number}"
                existing_checkpoint = session.query(Checkpoint).filter_by(page_url=page_url).first()
                if existing_checkpoint:
                    db_writer_logger.info(f"Page {page_url} already processed. Skipping.")
                    db_queue.task_done()
                    batch_ready_queue.put(None)
                    continue
                new_checkpoint = Checkpoint(page_url=page_url)
                session.add(new_checkpoint)
                session.commit()

                # Создание base_url
                base_urls = list(set([f"{urlparse(url).scheme}://{urlparse(url).netloc}" for url in image_urls]))
                base_url_str = base_urls[0]  # Предполагаем, что все изображения имеют одинаковый base_url
                base_url = BaseImageUrl(base_url=base_url_str)
                session.add(base_url)
                session.commit()

                # Создание батча
                batch = Batch(page_number=page_number)
                session.add(batch)
                session.commit()

                # Подготовка данных для вставки
                images_data = []
                image_filenames = []
                image_paths = []  # Новое
                image_urls_list = []

                for img_url in image_urls:
                    parsed_url = urlparse(img_url)
                    filename = os.path.basename(parsed_url.path)
                    path = os.path.dirname(parsed_url.path).lstrip('/')  # Extract the path
                    image = Image(base_url_id=base_url.id, path=path, filename=filename)
                    images_data.append(image)
                    image_filenames.append(filename)
                    image_paths.append(path)
                    image_urls_list.append(img_url)

                if not images_data:
                    db_writer_logger.info(f"No images to insert for page {page_url}.")
                    db_queue.task_done()
                    batch_ready_queue.put(None)
                    continue

                # Вставка изображений в базу данных
                session.add_all(images_data)
                session.commit()

                # Создание записей в batch_images
                batch_images_data = [BatchImage(batch_id=batch.id, image_id=img.id) for img in images_data]
                session.bulk_save_objects(batch_images_data)
                session.commit()

                # Передача информации о батче
                batch_info = {
                    'batch_id': batch.id,
                    'batch_dir': f"batch_{batch.id}",
                    'image_ids': [img.id for img in images_data],
                    'filenames': image_filenames,
                    'paths': image_paths,  # Новое
                    'image_urls': image_urls_list,
                }
                db_queue.task_done()
                db_writer_logger.info(f"Batch {batch.id} is ready and passed to downloader_thread.")
                batch_ready_queue.put(batch_info)
            elif task_type == 'mark_batch_processed':
                batch_id = data
                batch = session.query(Batch).filter_by(id=batch_id).first()
                if batch:
                    batch.processed = True
                    session.commit()
                db_queue.task_done()
        except Exception as e:
            session.rollback()
            db_writer_logger.error(f"Error in db_writer_thread: {e}")
            db_writer_logger.debug(traceback.format_exc())
            db_queue.task_done()
            batch_ready_queue.put(None)
    session.close()
