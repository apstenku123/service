# file: downloader.py
# directory: .
import threading
import time

import requests
# from queue import Queue
# from urllib.parse import urlparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup

from utils import configure_thread_logging, get_session_factory, get_engine
from models import ArchivedImage
# import traceback
import config  # Импортируем модуль конфигурации

def downloader_thread(page_queue, batch_queue, db_queue, batch_ready_queue, download_dir, download_threads, stats_collector, log_level, log_output, archive_enabled):
    # global MACHINE_ID
    # Set up logger for this function
    log_filename = f'logs/downloader/downloader_{config.MACHINE_ID}.log'
    downloader_logger = configure_thread_logging('downloader', log_filename, log_level, log_output)

    SessionFactory = get_session_factory(get_engine())
    session = SessionFactory()

    while True:
        page_info = page_queue.get()
        if page_info is None:
            page_queue.task_done()
            break  # Termination signal

        start_time = time.time()
        page_number = page_info['page_number']
        search_query = page_info.get('query')

        if search_query:
            page_url = f"http://camvideos.me/search/{search_query}?page={page_number}"
        else:
            page_url = f"http://camvideos.me/?page={page_number}"

        image_urls = process_page(page_url, stats_collector, log_level, log_output)

        if image_urls:
            # Send task to store batch in the database
            db_queue.put(('store_batch', (page_number, image_urls)))
            # Wait for the batch to be ready
            batch_info = batch_ready_queue.get()
            if batch_info is None:
                batch_ready_queue.task_done()
                downloader_logger.info(f"Batch for page {page_url} was not created.")
                page_queue.task_done()
                continue

            batch_id = batch_info['batch_id']
            batch_dir_name = batch_info['batch_dir']
            batch_dir = os.path.join(download_dir, batch_dir_name)
            os.makedirs(batch_dir, exist_ok=True)

            image_ids = batch_info['image_ids']
            filenames = batch_info['filenames']
            image_urls = batch_info['image_urls']

            # Map image IDs to filenames and URLs
            id_to_filename = dict(zip(image_ids, filenames))
            id_to_url = dict(zip(image_ids, image_urls))

            # For each image, check if it's archived
            images_to_download = []
            for img_id in image_ids:
                archived_image = session.query(ArchivedImage).filter_by(image_id=img_id).first()
                if archived_image and archive_enabled:
                    # Attempt to download from archive
                    archive_url = archived_image.archive_url
                    local_path = os.path.join(batch_dir, id_to_filename[img_id])
                    success = download_image(archive_url, local_path, downloader_logger, stats_collector)
                    if not success:
                        # If failed, add to download from original source
                        images_to_download.append(img_id)
                else:
                    images_to_download.append(img_id)

            # Now download images not found in archive
            if images_to_download:
                image_download_futures = []
                with ThreadPoolExecutor(max_workers=download_threads) as executor:
                    for img_id in images_to_download:
                        filename = id_to_filename[img_id]
                        img_url = id_to_url[img_id]
                        local_path = os.path.join(batch_dir, filename)
                        future = executor.submit(download_image, img_url, local_path, downloader_logger, stats_collector)
                        image_download_futures.append(future)

                    # Wait for all downloads to complete
                    for future in as_completed(image_download_futures):
                        result = future.result()
                        if not result:
                            downloader_logger.error(f"Error downloading image in batch {batch_id}")
            else:
                downloader_logger.info(f"All images for batch {batch_id} were downloaded from archive.")

            # Now that images are downloaded, add batch to processing queue
            batch_info = {
                'batch_id': batch_id,
                'batch_dir': batch_dir,
                'image_ids': image_ids,
                'filenames': filenames,
                'image_urls': image_urls,
            }
            batch_queue.put(batch_info)
            downloader_logger.info(f"Batch {batch_id} added to processing queue.")
            batch_ready_queue.task_done()
        else:
            downloader_logger.info(f"No images on page {page_url}")

        processing_time = time.time() - start_time
        stats_collector.add_batch_processing_time('downloader', processing_time)
        page_queue.task_done()
    session.close()

def process_page(page_url, stats_collector, log_level, log_output):
    # global MACHINE_ID
    # Set up logger for this function
    log_filename = f'logs/html_processor/html_processor_{config.MACHINE_ID}.log'
    html_logger = configure_thread_logging('html_processor', log_filename, log_level, log_output)

    html_logger.info(f"Processing page: {page_url}")
    try:
        response = requests.get(page_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        html_logger.error(f"Error loading page {page_url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract image URLs
    posts = soup.find_all('div', class_='post-container')
    image_urls = [post.find('img')['src'].replace('.th', '') for post in posts if post.find('img')]

    html_logger.info(f"Found {len(image_urls)} images on page {page_url}")
    return image_urls

def get_total_pages_for_query(query):
    url = f"http://camvideos.me/search/{query}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error loading page {url}: {e}")
        return 0

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

def download_image(image_url, local_path, downloader_logger, stats_collector):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            f.write(response.content)
        downloader_logger.info(f"Downloaded image: {image_url}")
        stats_collector.increment_files_downloaded()
        return True
    except Exception as e:
        downloader_logger.error(f"Error downloading {image_url}: {e}")
        return False
