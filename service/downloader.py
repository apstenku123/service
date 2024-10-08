import os
import threading
import time
import requests
import shutil
from concurrent.futures import ThreadPoolExecutor, wait
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from utils import configure_thread_logging, get_session_factory, get_engine
from models import ArchivedImage
import config  # Импортируем модуль конфигурации


def downloader_thread(page_queue, batch_queue, db_queue, batch_ready_queue, download_dir, download_threads,
                      stats_collector, log_level, log_output, archive_enabled, condition, MAX_BATCHES_ON_DISK):
    # Set up logger for this function
    log_filename = f'logs/downloader/downloader_{config.MACHINE_ID}.log'
    downloader_logger = configure_thread_logging('downloader', log_filename, log_level, log_output)

    SessionFactory = get_session_factory(get_engine())
    session = SessionFactory()

    # Создаем ThreadPoolExecutor один раз
    executor = ThreadPoolExecutor(max_workers=download_threads)

    while True:
        page_info = page_queue.get()
        if page_info is None:
            page_queue.task_done()
            break  # Termination signal

        try:
            start_time = time.time()
            page_number = page_info['page_number']
            search_query = page_info.get('query')

            if search_query:
                page_url = f"http://camvideos.me/search/{search_query}?page={page_number}"
            else:
                page_url = f"http://camvideos.me/?page={page_number}"

            image_urls = process_page(page_url, stats_collector, log_level, log_output)

            if not image_urls:
                downloader_logger.info(f"No images on page {page_url}")
                page_queue.task_done()
                continue

            # **Отправляем задачу на запись батча в базу данных**
            db_queue.put(('store_batch', (page_number, image_urls)))

            # Wait until current_batches_on_disk < MAX_BATCHES_ON_DISK
            with condition:
                while config.current_batches_on_disk >= MAX_BATCHES_ON_DISK:
                    condition.wait()
                config.current_batches_on_disk += 1  # Increment the counter

            # **Начинаем загрузку изображений сразу, без ожидания записи батча в базу данных**

            # Генерируем имена файлов на основе URL
            filenames = [os.path.basename(urlparse(url).path) for url in image_urls]

            # Создаем директорию для батча
            batch_dir_name = f"batch_{page_number}"
            batch_dir = os.path.join(download_dir, batch_dir_name)
            os.makedirs(batch_dir, exist_ok=True)

            # Map filenames to URLs
            filename_to_url = dict(zip(filenames, image_urls))

            # **Оптимизируем запрос к базе данных для ArchivedImage**

            if archive_enabled:
                # Получаем все ArchivedImage для текущего батча по именам файлов одним запросом
                try:
                    archived_images = session.query(ArchivedImage).filter(ArchivedImage.filename.in_(filenames)).all()
                    archived_filenames = {img.filename for img in archived_images}
                    filename_to_archive_url = {img.filename: img.archive_url for img in archived_images}
                except Exception as e:
                    downloader_logger.error(f"Database error while querying ArchivedImage: {e}")
                    archived_filenames = set()
                    filename_to_archive_url = {}
            else:
                archived_filenames = set()
                filename_to_archive_url = {}

            images_to_download = []
            for filename in filenames:
                if archive_enabled and filename in archived_filenames:
                    # Изображение есть в архиве
                    archive_url = filename_to_archive_url[filename]
                    local_path = os.path.join(batch_dir, filename)
                    success = download_image(archive_url, local_path, downloader_logger, stats_collector)
                    if not success:
                        # Если загрузка из архива не удалась, добавляем в список для загрузки из источника
                        images_to_download.append(filename)
                else:
                    images_to_download.append(filename)

            # Теперь загружаем изображения, не найденные в архиве
            if images_to_download:
                futures = []

                for filename in images_to_download:
                    img_url = filename_to_url[filename]
                    local_path = os.path.join(batch_dir, filename)
                    future = executor.submit(download_image, img_url, local_path, downloader_logger, stats_collector)
                    futures.append(future)

                # Ждем завершения всех загрузок для данного батча
                wait(futures)

                # Проверяем результаты загрузок
                failed_downloads = []
                for future in futures:
                    result = future.result()
                    if not result:
                        failed_downloads.append(future)

                if failed_downloads:
                    downloader_logger.warning(f"Batch for page {page_number} has failed downloads.")
                    # Решаем, что делать с неудачными загрузками
                    # Например, можно удалить батч и пропустить его
                    # Или продолжить обработку с имеющимися файлами
                else:
                    downloader_logger.info(f"All images for batch from page {page_number} downloaded successfully.")

            else:
                downloader_logger.info(f"All images for batch from page {page_number} were downloaded from archive.")

            # **Ждём, пока батч будет записан в базу данных**
            batch_info = batch_ready_queue.get()
            if batch_info is None:
                batch_ready_queue.task_done()
                downloader_logger.error(f"Failed to create batch for page {page_url}.")
                # Удаляем загруженные файлы и директорию
                shutil.rmtree(batch_dir, ignore_errors=True)
                # Уменьшаем счетчик текущих батчей на диске
                with condition:
                    config.current_batches_on_disk -= 1
                    condition.notify()
                page_queue.task_done()
                continue

            batch_id = batch_info['batch_id']
            image_ids = batch_info['image_ids']
            # Обновляем batch_info с актуальной информацией
            batch_info.update({
                'batch_dir': batch_dir,
                'filenames': filenames,
                'image_urls': image_urls,
            })

            # Теперь, когда батч готов и все изображения загружены, добавляем его в очередь на обработку
            batch_queue.put(batch_info)
            downloader_logger.info(f"Batch {batch_id} added to processing queue.")

            batch_ready_queue.task_done()

        except Exception as e:
            downloader_logger.error(f"Error processing page {page_number}: {e}", exc_info=True)
            # Удаляем загруженные файлы и директорию в случае ошибки
            if 'batch_dir' in locals():
                shutil.rmtree(batch_dir, ignore_errors=True)
            # Уменьшаем счетчик текущих батчей на диске
            with condition:
                config.current_batches_on_disk -= 1
                condition.notify()
        finally:
            processing_time = time.time() - start_time
            stats_collector.add_batch_processing_time('downloader', processing_time)
            page_queue.task_done()

    # Ожидаем завершения всех задач перед завершением
    executor.shutdown(wait=True)
    session.close()


def process_page(page_url, stats_collector, log_level, log_output):
    # Set up logger for this function
    log_filename = f'logs/html_processor/html_processor_{config.MACHINE_ID}.log'
    html_logger = configure_thread_logging('html_processor', log_filename, log_level, log_output)

    html_logger.info(f"Processing page: {page_url}")
    try:
        response = requests.get(page_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        html_logger.error(f"Error loading page {page_url}: {e}")
        return []

    try:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract image URLs
        posts = soup.find_all('div', class_='post-container')
        image_urls = [post.find('img')['src'].replace('.th', '') for post in posts if post.find('img')]

        html_logger.info(f"Found {len(image_urls)} images on page {page_url}")
        return image_urls
    except Exception as e:
        html_logger.error(f"Error parsing page {page_url}: {e}", exc_info=True)
        return []


def get_total_pages_for_query(query):
    url = f"http://camvideos.me/search/{query}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error loading page {url}: {e}")
        return 0

    try:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Ищем элемент с классом 'numberofpages'
        numberofpages_element = soup.find('a', class_='numberofpages')
        if numberofpages_element:
            total_pages_text = numberofpages_element.text.strip('..')
            try:
                total_pages = int(total_pages_text)
                return total_pages
            except ValueError:
                print(f"Could not parse total pages from text: {total_pages_text}")
                return 1
        else:
            # Если элемент не найден, проверяем наличие постов
            posts = soup.find_all('div', class_='post-container')
            if posts:
                return 1
            else:
                return 0
    except Exception as e:
        print(f"Error parsing page {url}: {e}")
        return 0


def download_image(image_url, local_path, downloader_logger, stats_collector):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            f.write(response.content)
        downloader_logger.info(f"Downloaded image: {image_url}")
        stats_collector.increment_files_downloaded()
        return True
    except requests.RequestException as e:
        downloader_logger.error(f"Error downloading {image_url}: {e}")
        return False
    except IOError as e:
        downloader_logger.error(f"Error saving image {image_url} to {local_path}: {e}")
        return False
