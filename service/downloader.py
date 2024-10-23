# file: downloader.py
# directory: .

import os
import threading
import time
import requests
import shutil
from concurrent.futures import ThreadPoolExecutor, wait
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from utils import configure_thread_logging, get_session_factory, get_engine
from models import ArchivedImage, Image, BatchImage, Batch
import config  # Import configuration module


def downloader_thread(page_queue, batch_queue, db_queue, batch_ready_queue, download_dir, download_threads,
                      stats_collector, log_level, log_output, archive_enabled, condition, MAX_BATCHES_ON_DISK):
    # Set up logger for this function
    log_filename = f'logs/downloader/downloader_{config.MACHINE_ID}.log'
    downloader_logger = configure_thread_logging('downloader', log_filename, log_level, log_output)

    SessionFactory = get_session_factory(get_engine())
    session = SessionFactory()

    # Create ThreadPoolExecutor once
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

            # Send task to store batch in database
            db_queue.put(('store_batch', (page_number, image_urls)))

            # Wait until current_batches_on_disk < MAX_BATCHES_ON_DISK
            with condition:
                while config.current_batches_on_disk >= MAX_BATCHES_ON_DISK:
                    condition.wait()
                config.current_batches_on_disk += 1  # Increment the counter

            # Wait for batch_info from db_writer
            batch_info = batch_ready_queue.get()
            if batch_info is None:
                batch_ready_queue.task_done()
                downloader_logger.error(f"Failed to create batch for page {page_url}.")
                # Decrease counter of current batches on disk
                with condition:
                    config.current_batches_on_disk -= 1
                    condition.notify()
                page_queue.task_done()
                continue

            batch_id = batch_info['batch_id']
            image_ids = batch_info['image_ids']
            filenames = batch_info['filenames']
            image_urls = batch_info['image_urls']
            image_paths = batch_info['paths']  # Ensure 'paths' are included

            filename_to_id = dict(zip(filenames, image_ids))
            id_to_url = dict(zip(image_ids, image_urls))

            # Update batch_info with actual information
            batch_dir_name = f"batch_{batch_id}"
            batch_dir = os.path.join(download_dir, batch_dir_name)
            os.makedirs(batch_dir, exist_ok=True)
            batch_info['batch_dir'] = batch_dir

            # Map filenames to URLs
            filename_to_url = dict(zip(filenames, image_urls))

            # Optimize database query for ArchivedImage
            if archive_enabled:
                # Get all ArchivedImage for current batch by filenames in one query
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
            failed_downloads = []  # List to keep track of failed downloads

            for filename in filenames:
                if archive_enabled and filename in archived_filenames:
                    # Image is in archive
                    archive_url = filename_to_archive_url[filename]
                    local_path = os.path.join(batch_dir, filename)
                    success = download_image(archive_url, local_path, downloader_logger, stats_collector)
                    if not success:
                        # If download from archive failed, add to list for downloading from source
                        images_to_download.append(filename)
                else:
                    images_to_download.append(filename)

            # Now download images not found in archive or failed to download from archive
            if images_to_download:
                futures = []

                for filename in images_to_download:
                    img_url = filename_to_url[filename]
                    local_path = os.path.join(batch_dir, filename)
                    future = executor.submit(download_image, img_url, local_path, downloader_logger, stats_collector)
                    future.filename = filename  # Attach filename to future for later identification
                    futures.append(future)

                # Wait for all downloads to complete for this batch
                wait(futures)

                # Check download results
                # Check download results
                for future in futures:
                    try:
                        result = future.result()
                        if not result:
                            failed_downloads.append(future.filename)
                    except Exception as e:
                        downloader_logger.info(f"Exception raised during downloading {future.filename}: {e}",
                                                exc_info=True)
                        failed_downloads.append(future.filename)

            # Handle failed downloads
            if failed_downloads:
                downloader_logger.info(f"Failed to download images: {failed_downloads}")
                # Remove image records from database for failed downloads
                try:
                    # Remove from BatchImage
                    session.query(BatchImage).filter(BatchImage.batch_id == batch_id,
                                                     BatchImage.image_id.in_(
                                                         [filename_to_id[fn] for fn in failed_downloads])
                                                     ).delete(synchronize_session=False)
                    # Remove from Image
                    session.query(Image).filter(Image.id.in_([filename_to_id[fn] for fn in failed_downloads])
                                                ).delete(synchronize_session=False)
                    session.commit()
                    downloader_logger.info(f"Removed failed images from database: {failed_downloads}")
                except Exception as e:
                    session.rollback()
                    downloader_logger.error(f"Error removing failed images from database: {e}", exc_info=True)
            else:
                downloader_logger.info(f"All images for batch {batch_id} downloaded successfully.")

            # Now check if there are any images left in the batch after removing failed downloads
            remaining_images = [fn for fn in filenames if fn not in failed_downloads]
            if not remaining_images:
                downloader_logger.warning(f"No images left in batch {batch_id} after removing failed downloads.")
                # Remove batch from database
                try:
                    # Remove from BatchImage
                    session.query(BatchImage).filter(BatchImage.batch_id == batch_id).delete(synchronize_session=False)
                    # Remove batch
                    session.query(Batch).filter(Batch.id == batch_id).delete(synchronize_session=False)
                    session.commit()
                    downloader_logger.info(f"Removed batch {batch_id} from database.")
                except Exception as e:
                    session.rollback()
                    downloader_logger.error(f"Error removing batch {batch_id} from database: {e}", exc_info=True)
                # Delete batch directory
                shutil.rmtree(batch_dir, ignore_errors=True)
                # Decrease counter of current batches on disk
                with condition:
                    config.current_batches_on_disk -= 1
                    condition.notify()
                # Inform the processor to skip this batch
                batch_ready_queue.task_done()
                # Since we are not adding to batch_queue, we need to task_done for batch_queue as well
                batch_queue.task_done()
                page_queue.task_done()
                continue
            else:    # Update batch_info to reflect remaining images
                batch_info['filenames'] = remaining_images
                batch_info['image_ids'] = [filename_to_id[fn] for fn in remaining_images]
                batch_info['image_urls'] = [filename_to_url[fn] for fn in remaining_images]
                batch_info['paths'] = [batch_info['paths'][filenames.index(fn)] for fn in remaining_images]

                # **Rebuild filename_to_id and filename_to_url mappings**
                filename_to_id = {fn: filename_to_id[fn] for fn in remaining_images}
                filename_to_url = {fn: filename_to_url[fn] for fn in remaining_images}

                # Now, when batch is ready and all images are downloaded, add it to processing queue
                batch_queue.put(batch_info)
                downloader_logger.info(f"Batch {batch_id} added to processing queue.")

                batch_ready_queue.task_done()

        except Exception as e:
            downloader_logger.error(f"Error processing page {page_number}: {e}", exc_info=True)
            # Delete downloaded files and directory in case of error
            if 'batch_dir' in locals():
                shutil.rmtree(batch_dir, ignore_errors=True)
            # Decrease counter of current batches on disk
            with condition:
                config.current_batches_on_disk -= 1
                condition.notify()
        finally:
            processing_time = time.time() - start_time
            stats_collector.add_batch_processing_time('downloader', processing_time)
            page_queue.task_done()

    # Wait for all tasks to complete before shutting down
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

        # Look for element with class 'numberofpages'
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
            # If element not found, check for posts
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
        downloader_logger.info(f"Error downloading {image_url}: {e}")
        return False
    except IOError as e:
        downloader_logger.error(f"Error saving image {image_url} to {local_path}: {e}")
        return False
