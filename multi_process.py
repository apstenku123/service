import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import argparse
import logging
import threading
from queue import Queue
from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, DateTime, UniqueConstraint, Index
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import IntegrityError
from datetime import datetime
import torch
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import traceback
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import socket
from logging.handlers import RotatingFileHandler

# Import additional libraries for archiving
import boto3
from azure.storage.blob import BlobServiceClient
import ftplib
import paramiko

# Base class for SQLAlchemy ORM
Base = declarative_base()

# ORM models with indexes defined
class BaseImageUrl(Base):
    __tablename__ = 'base_image_urls'
    id = Column(Integer, primary_key=True)
    base_url = Column(String, nullable=False)
    images = relationship("Image", back_populates="base_image_url")

    __table_args__ = (
        Index('idx_base_image_urls_base_url', 'base_url'),
    )

class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True)
    base_url_id = Column(Integer, ForeignKey('base_image_urls.id'), nullable=False)
    filename = Column(String, nullable=False)
    processed = Column(Boolean, default=False)
    base_image_url = relationship("BaseImageUrl", back_populates="images")
    embeddings = relationship("ImageEmbedding", back_populates="image")
    archived_image = relationship("ArchivedImage", back_populates="image", uselist=False)

    __table_args__ = (
        UniqueConstraint('base_url_id', 'filename', name='_base_image_uc'),
        Index('idx_images_base_url_id', 'base_url_id'),
        Index('idx_images_processed', 'processed'),
    )

class Batch(Base):
    __tablename__ = 'batches'
    id = Column(Integer, primary_key=True)
    page_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.now())
    processed = Column(Boolean, default=False)
    images = relationship("BatchImage", back_populates="batch")

    __table_args__ = (
        UniqueConstraint('page_number', name='_page_number_uc'),
        Index('idx_batches_processed', 'processed'),
    )

class BatchImage(Base):
    __tablename__ = 'batch_images'
    batch_id = Column(Integer, ForeignKey('batches.id'), primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), primary_key=True)
    batch = relationship("Batch", back_populates="images")
    image = relationship("Image")

    __table_args__ = (
        Index('idx_batch_images_batch_id', 'batch_id'),
        Index('idx_batch_images_image_id', 'image_id'),
    )

class Checkpoint(Base):
    __tablename__ = 'checkpoints'
    id = Column(Integer, primary_key=True)
    page_url = Column(String, unique=True, nullable=False)

    __table_args__ = (
        Index('idx_checkpoints_page_url', 'page_url'),
    )

class ImageEmbedding(Base):
    __tablename__ = 'image_embeddings'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'))
    filename = Column(String, nullable=False)
    embedding = Column(String, nullable=False)  # Stored as string for simplicity
    image = relationship("Image", back_populates="embeddings")

    __table_args__ = (
        Index('idx_image_embeddings_image_id', 'image_id'),
    )

# New models for host and log data
class HostLog(Base):
    __tablename__ = 'host_logs'
    id = Column(Integer, primary_key=True)
    host_name = Column(String, nullable=False)
    function_name = Column(String, nullable=False)
    log_file = Column(String, nullable=False)
    batches = relationship("BatchLog", back_populates="host_log")

    __table_args__ = (
        Index('idx_host_logs_host_name', 'host_name'),
        Index('idx_host_logs_function_name', 'function_name'),
    )

class BatchLog(Base):
    __tablename__ = 'batch_logs'
    id = Column(Integer, primary_key=True)
    batch_id = Column(Integer, ForeignKey('batches.id'), nullable=False)
    host_log_id = Column(Integer, ForeignKey('host_logs.id'), nullable=False)
    host_log = relationship("HostLog", back_populates="batches")

    __table_args__ = (
        Index('idx_batch_logs_batch_id', 'batch_id'),
    )

# New model for archived images
class ArchivedImage(Base):
    __tablename__ = 'archived_images'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False, unique=True)
    archive_url = Column(String, nullable=False)
    image = relationship("Image", back_populates="archived_image")

    __table_args__ = (
        Index('idx_archived_images_image_id', 'image_id'),
        Index('idx_archived_images_archive_url', 'archive_url'),
    )

# Function to connect to the database
def get_engine():
    DB_HOST = os.environ.get('DB_HOST')
    DB_NAME = os.environ.get('DB_NAME')
    DB_USER = os.environ.get('DB_USER')
    DB_PASSWORD = os.environ.get('DB_PASSWORD')

    DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
    engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
    return engine

# Create session factory
def get_session_factory(engine):
    return sessionmaker(bind=engine)

# Function to set up the database
def setup_database(engine):
    Base.metadata.create_all(engine)

# Statistics collector class
class StatsCollector:
    def __init__(self):
        self.lock = threading.Lock()
        # Resettable stats
        self.files_downloaded = 0
        self.faces_found = 0
        self.embeddings_uploaded = 0
        self.batch_processing_times = {'downloader': [], 'embedding_processor': [], 'embeddings_writer': []}
        self.last_reset_time = time.time()
        # Global stats
        self.total_files_downloaded = 0
        self.total_faces_found = 0
        self.total_embeddings_uploaded = 0
        self.total_images_processed = 0
        self.total_batches_processed = 0
        self.images_with_faces = 0
        self.images_without_faces = 0
        self.start_time = time.time()

    def increment_files_downloaded(self, count=1):
        with self.lock:
            self.files_downloaded += count
            self.total_files_downloaded += count

    def increment_faces_found(self, count=1):
        with self.lock:
            self.faces_found += count
            self.total_faces_found += count

    def increment_embeddings_uploaded(self, count=1):
        with self.lock:
            self.embeddings_uploaded += count
            self.total_embeddings_uploaded += count

    def increment_images_processed(self, count=1):
        with self.lock:
            self.total_images_processed += count

    def increment_batches_processed(self, count=1):
        with self.lock:
            self.total_batches_processed += count

    def increment_images_with_faces(self, count=1):
        with self.lock:
            self.images_with_faces += count

    def increment_images_without_faces(self, count=1):
        with self.lock:
            self.images_without_faces += count

    def add_batch_processing_time(self, thread_name, processing_time):
        with self.lock:
            self.batch_processing_times[thread_name].append(processing_time)

    def reset(self):
        with self.lock:
            stats = {
                # Interval stats
                'files_downloaded': self.files_downloaded,
                'faces_found': self.faces_found,
                'embeddings_uploaded': self.embeddings_uploaded,
                'batch_processing_times': self.batch_processing_times.copy(),
                'elapsed_time': time.time() - self.last_reset_time,
                # Global stats
                'total_files_downloaded': self.total_files_downloaded,
                'total_faces_found': self.total_faces_found,
                'total_embeddings_uploaded': self.total_embeddings_uploaded,
                'total_images_processed': self.total_images_processed,
                'total_batches_processed': self.total_batches_processed,
                'images_with_faces': self.images_with_faces,
                'images_without_faces': self.images_without_faces,
                'start_time': self.start_time,
            }
            # Reset interval stats
            self.files_downloaded = 0
            self.faces_found = 0
            self.embeddings_uploaded = 0
            self.batch_processing_times = {'downloader': [], 'embedding_processor': [], 'embeddings_writer': []}
            self.last_reset_time = time.time()
        return stats

# Function to preprocess the image
def preprocess_image(image_path):
    """
    Loads and processes an image.
    """
    image = cv2.imread(image_path)
    if image is not None:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image
    return None

# Function to configure logging for a thread
def configure_thread_logging(logger_name, log_filename, log_level, log_output):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    handlers = []
    if log_output in ('file', 'both'):
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        file_handler = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    if log_output in ('console', 'both'):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    # Remove existing handlers
    logger.handlers = []
    for handler in handlers:
        logger.addHandler(handler)

    return logger

# Function to process a page and extract image URLs
def process_page(page_url, stats_collector, log_level, log_output):
    global MACHINE_ID
    # Set up logger for this function
    log_filename = os.path.join('logs', 'html_processor', f'html_processor_{MACHINE_ID}.log')
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

# Statistics logging thread
def stats_logger_thread(stats_collector, interval, stop_event, log_level, log_output, total_pages_to_process, page_queue, batch_queue, embeddings_queue, db_queue):
    global MACHINE_ID
    stats_logger = configure_thread_logging('stats_logger', f'logs/stats_logger/stats_logger_{MACHINE_ID}.log', log_level, log_output)

    while not stop_event.is_set():
        time.sleep(interval)
        stats = stats_collector.reset()
        # Calculate average processing times
        avg_times = {}
        for thread_name, times in stats['batch_processing_times'].items():
            if times:
                avg_times[thread_name] = sum(times) / len(times)
            else:
                avg_times[thread_name] = 0

        total_images_processed = stats['total_images_processed']
        total_faces_found = stats['total_faces_found']
        images_with_faces = stats['images_with_faces']
        images_without_faces = stats['images_without_faces']
        average_faces_per_image = total_faces_found / total_images_processed if total_images_processed > 0 else 0

        # Estimate remaining time
        elapsed_time = time.time() - stats['start_time']
        batches_processed = stats['total_batches_processed']
        pages_processed = batches_processed  # Assuming one batch per page
        pages_remaining = total_pages_to_process - pages_processed
        avg_page_time = (elapsed_time / pages_processed) if pages_processed > 0 else 0
        estimated_time_remaining = avg_page_time * pages_remaining

        message = (
            f"Total Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}\n"
            f"Interval Stats:\n"
            f"  Files Downloaded: {stats['files_downloaded']}\n"
            f"  Faces Found: {stats['faces_found']}\n"
            f"  Embeddings Uploaded: {stats['embeddings_uploaded']}\n"
            f"Average Batch Processing Times:\n"
        )
        for thread_name, avg_time in avg_times.items():
            message += f"  {thread_name}: {avg_time:.2f}s\n"

        message += (
            f"Global Stats:\n"
            f"  Total Files Downloaded: {stats['total_files_downloaded']}\n"
            f"  Total Images Processed: {total_images_processed}\n"
            f"  Total Faces Found: {total_faces_found}\n"
            f"  Images with Faces: {images_with_faces}\n"
            f"  Images without Faces: {images_without_faces}\n"
            f"  Average Faces per Image: {average_faces_per_image:.2f}\n"
            f"  Total Batches Processed: {batches_processed}\n"
            f"Queue Sizes:\n"
            f"  Page Queue: {page_queue.qsize()}\n"
            f"  Batch Queue: {batch_queue.qsize()}\n"
            f"  Embeddings Queue: {embeddings_queue.qsize()}\n"
            f"  DB Queue: {db_queue.qsize()}\n"
            f"Estimated Time Remaining: {time.strftime('%H:%M:%S', time.gmtime(estimated_time_remaining))}\n"
        )

        stats_logger.info(message)
        print(message)

# Database writer thread
def db_writer_thread(db_queue, batch_ready_queue, engine, stats_collector, log_level, log_output):
    global MACHINE_ID
    # Set up logger for this function
    log_filename = os.path.join('logs', 'db_writer', f'db_writer_{MACHINE_ID}.log')
    db_writer_logger = configure_thread_logging('db_writer', log_filename, log_level, log_output)

    SessionFactory = get_session_factory(engine)
    session = SessionFactory()
    host_name = socket.gethostname()

    # Check if HostLog exists
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
                # Handle checkpoint
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

                # Create a new base_url for each batch
                base_urls = list(set([f"{urlparse(url).scheme}://{urlparse(url).netloc}" for url in image_urls]))
                base_url_str = base_urls[0]  # Assuming all images have the same base URL
                base_url = BaseImageUrl(base_url=base_url_str)
                session.add(base_url)
                session.commit()

                # Create batch
                batch = Batch(page_number=page_number)
                session.add(batch)
                session.commit()

                # Prepare data for insertion
                images_data = []
                image_filenames = []
                image_urls_list = []
                for img_url in image_urls:
                    filename = os.path.basename(urlparse(img_url).path)
                    image = Image(base_url_id=base_url.id, filename=filename)
                    images_data.append(image)
                    image_filenames.append(filename)
                    image_urls_list.append(img_url)

                if not images_data:
                    db_writer_logger.info(f"No images to insert for page {page_url}.")
                    db_queue.task_done()
                    batch_ready_queue.put(None)
                    continue

                # Insert images into the database
                session.add_all(images_data)
                session.commit()

                # Create entries in batch_images
                batch_images_data = [BatchImage(batch_id=batch.id, image_id=img.id) for img in images_data]
                session.bulk_save_objects(batch_images_data)
                session.commit()

                # Pass batch information
                batch_info = {
                    'batch_id': batch.id,
                    'batch_dir': f"batch_{batch.id}",
                    'image_ids': [img.id for img in images_data],
                    'filenames': image_filenames,
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

# Downloader thread
def downloader_thread(page_queue, batch_queue, db_queue, batch_ready_queue, download_dir, download_threads, stats_collector, log_level, log_output, archive_enabled):
    global MACHINE_ID
    # Set up logger for this function
    log_filename = os.path.join('logs', 'downloader', f'downloader_{MACHINE_ID}.log')
    downloader_logger = configure_thread_logging('downloader', log_filename, log_level, log_output)

    SessionFactory = get_session_factory(get_engine())
    session = SessionFactory()

    while True:
        page_info = page_queue.get()
        if page_info is None:
            page_queue.task_done()
            break  # Termination signal

        start_time = time.time()
        page_number = page_info
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


# Processing thread
def processing_thread(batch_queue, embeddings_queue, archive_queue, model, mtcnn, device, engine, batch_size, report_dir, stats_collector, log_level, log_output, images_without_faces_log_file):
    global MACHINE_ID
    # Set up logger for this function
    log_filename = os.path.join('logs', 'embedding_processor', f'embedding_processor_{MACHINE_ID}.log')
    embedding_processor_logger = configure_thread_logging('embedding_processor', log_filename, log_level, log_output)

    cycle_iterator = 0
    # Get host name
    host_name = socket.gethostname()

    # Initialize database session
    SessionFactory = get_session_factory(engine)
    session = SessionFactory()

    # Check if HostLog exists
    existing_host_log = session.query(HostLog).filter_by(host_name=host_name, function_name='embedding_processor', log_file=log_filename).first()
    if not existing_host_log:
        host_log = HostLog(host_name=host_name, function_name='embedding_processor', log_file=log_filename)
        session.add(host_log)
        session.commit()
    else:
        host_log = existing_host_log

    while True:
        batch_info = batch_queue.get()
        if batch_info is None:
            batch_queue.task_done()
            break  # Termination signal

        start_time = time.time()
        batch_id = batch_info['batch_id']
        batch_dir = batch_info['batch_dir']
        image_ids = batch_info['image_ids']
        filenames = batch_info['filenames']
        image_urls = batch_info['image_urls']

        # Create mapping from filename to image_url and image_id
        filename_to_url = dict(zip(filenames, image_urls))
        filename_to_id = dict(zip(filenames, image_ids))

        # Check if batch is already processed
        batch = session.query(Batch).filter_by(id=batch_id).first()
        if batch.processed:
            embedding_processor_logger.info(f"Batch {batch_id} is already processed. Skipping.")
            batch_queue.task_done()
            continue

        embedding_processor_logger.info(f"Starting processing of batch {batch_id}")
        embeddings_data = []
        try:
            images = session.query(Image).join(BatchImage).filter(BatchImage.batch_id == batch_id).all()
            image_paths = []
            image_ids = []
            filenames = []
            for img in images:
                local_path = os.path.join(batch_dir, img.filename)
                if not os.path.exists(local_path):
                    embedding_processor_logger.error(f"File not found: {local_path}")
                    continue
                image_paths.append(local_path)
                image_ids.append(img.id)
                filenames.append(img.filename)

            # Preprocess images (load and convert)
            images_data_list = []
            valid_image_ids = []
            valid_filenames = []
            image_sizes = []
            for idx, path in enumerate(image_paths):
                img_rgb = preprocess_image(path)
                if img_rgb is not None:
                    images_data_list.append(img_rgb)
                    valid_image_ids.append(image_ids[idx])
                    valid_filenames.append(filenames[idx])
                    image_sizes.append(img_rgb.size)  # Store image size
                else:
                    embedding_processor_logger.error(f"Failed to load image: {path}")

            total_images = len(images_data_list)
            total_faces = 0
            images_with_faces = 0
            images_without_faces = 0

            # Group images by size
            from collections import defaultdict
            size_to_indices = defaultdict(list)
            for idx, size in enumerate(image_sizes):
                size_to_indices[size].append(idx)

            for size, indices in size_to_indices.items():
                embedding_processor_logger.info(f"Processing group of images with size {size}, count: {len(indices)}")
                # Process each group of images with the same size
                group_images_data = [images_data_list[idx] for idx in indices]
                group_image_ids = [valid_image_ids[idx] for idx in indices]
                group_filenames = [valid_filenames[idx] for idx in indices]

                # Split group into batches
                for i in range(0, len(group_images_data), batch_size):
                    batch_imgs = group_images_data[i:i + batch_size]
                    batch_ids = group_image_ids[i:i + batch_size]
                    batch_filenames = group_filenames[i:i + batch_size]

                    # Detect faces in batch
                    try:
                        faces_batch = mtcnn(batch_imgs)
                    except Exception as e:
                        embedding_processor_logger.error(f"Face detection error in batch: {e}")
                        faces_batch = [None] * len(batch_imgs)

                    batch_faces = []
                    face_image_ids = []
                    face_filenames = []
                    for idx_in_batch, faces in enumerate(faces_batch):
                        if faces is not None:
                            num_faces = faces.shape[0] if len(faces.shape) == 4 else 1
                            total_faces += num_faces
                            images_with_faces += 1
                            stats_collector.increment_faces_found(num_faces)
                            stats_collector.increment_images_with_faces()
                            # If multiple faces in image
                            if len(faces.shape) == 4:
                                for j in range(num_faces):
                                    face_tensor = faces[j].to(device)
                                    batch_faces.append(face_tensor)
                                    face_image_ids.append(batch_ids[idx_in_batch])
                                    face_filenames.append(batch_filenames[idx_in_batch])
                            else:
                                face_tensor = faces.to(device)
                                batch_faces.append(face_tensor)
                                face_image_ids.append(batch_ids[idx_in_batch])
                                face_filenames.append(batch_filenames[idx_in_batch])
                        else:
                            images_without_faces += 1
                            stats_collector.increment_images_without_faces()
                            embedding_processor_logger.info(f"No faces detected in image: {batch_filenames[idx_in_batch]}")

                            # Write full URL to log file
                            full_url = filename_to_url[batch_filenames[idx_in_batch]]
                            images_without_faces_log_file.write(f"{full_url}\n")

                    if not batch_faces:
                        continue

                    # Process embeddings in batch
                    faces_tensor = torch.stack(batch_faces)
                    with torch.no_grad():
                        embeddings = model(faces_tensor).cpu().tolist()

                    # Collect embeddings
                    for img_id, filename, embedding in zip(face_image_ids, face_filenames, embeddings):
                        embeddings_data.append({'image_id': img_id, 'filename': filename, 'embedding': embedding})

            # Pass data for saving to database
            embeddings_queue.put((batch_id, embeddings_data))
            embedding_processor_logger.info(f"Embeddings data for batch {batch_id} added to embeddings_queue.")

            # Generate report for batch
            report = {
                'batch_id': batch_id,
                'total_images': total_images,
                'total_faces': total_faces,
                'images_with_faces': images_with_faces,
                'images_without_faces': images_without_faces
            }
            report_filename = os.path.join(report_dir, f"batch_{batch_id}_report.txt")
            with open(report_filename, 'w') as f:
                f.write(f"Batch ID: {batch_id}\n")
                f.write(f"Total images processed: {total_images}\n")
                f.write(f"Total faces detected: {total_faces}\n")
                f.write(f"Images with faces: {images_with_faces}\n")
                f.write(f"Images without faces: {images_without_faces}\n")
            embedding_processor_logger.info(f"Report for batch {batch_id} saved to {report_filename}")

            embedding_processor_logger.info(f"Batch {batch_id} processed and passed for embedding saving.")

            # Mark images as processed
            session.query(Image).filter(Image.id.in_(valid_image_ids)).update({"processed": True}, synchronize_session=False)

            # Update statistics
            stats_collector.increment_images_processed(len(valid_image_ids))
            stats_collector.increment_batches_processed()

            # Associate batch with log file
            batch_log = BatchLog(batch_id=batch_id, host_log_id=host_log.id)
            session.add(batch_log)
            session.commit()

            # After processing, add batch to archive queue
            archive_info = {
                'batch_id': batch_id,
                'batch_dir': batch_dir,
                'filenames': filenames,
                'image_ids': image_ids,
                'filename_to_id': filename_to_id,
                'filename_to_url': filename_to_url,
            }
            archive_queue.put(archive_info)
            embedding_processor_logger.info(f"Batch {batch_id} added to archive queue.")

            batch_queue.task_done()
        except Exception as e:
            session.rollback()
            embedding_processor_logger.error(f"Error processing batch {batch_id}: {e}")
            embedding_processor_logger.debug(traceback.format_exc())
            batch_queue.task_done()

        processing_time = time.time() - start_time
        stats_collector.add_batch_processing_time('embedding_processor', processing_time)

        cycle_iterator += 1
        if cycle_iterator % 16 == 0:
            torch.cuda.empty_cache()

        time.sleep(1)  # Small delay to reduce load
    session.close()

# Embeddings writer thread
def embeddings_writer_thread(embeddings_queue, db_queue, engine, stats_collector, log_level, log_output):
    global MACHINE_ID
    # Set up logger for this function
    log_filename = os.path.join('logs', 'embeddings_writer', f'embeddings_writer_{MACHINE_ID}.log')
    embeddings_writer_logger = configure_thread_logging('embeddings_writer', log_filename, log_level, log_output)

    SessionFactory = get_session_factory(engine)
    session = SessionFactory()
    while True:
        embeddings_info = embeddings_queue.get()
        embeddings_writer_logger.info("Received embeddings_info from queue.")
        if embeddings_info is None:
            embeddings_writer_logger.info("Termination signal received. Exiting embeddings_writer_thread.")
            break  # Termination signal

        start_time = time.time()
        batch_id, embeddings_data = embeddings_info
        embeddings_writer_logger.info(f"Starting to save embeddings for batch {batch_id}")
        try:
            if not embeddings_data:
                embeddings_writer_logger.info(f"No embeddings to save in batch {batch_id}")
                embeddings_queue.task_done()
                continue

            # Save embeddings to database in one transaction
            embeddings_objects = []
            for data in embeddings_data:
                embedding_str = ','.join(map(str, data['embedding']))
                embedding = ImageEmbedding(image_id=data['image_id'], filename=data['filename'], embedding=embedding_str)
                embeddings_objects.append(embedding)
            session.bulk_save_objects(embeddings_objects)
            session.commit()
            embeddings_writer_logger.info(f"Embeddings for batch {batch_id} committed to database.")

            # Update statistics
            stats_collector.increment_embeddings_uploaded(len(embeddings_objects))

            # Mark batch as processed
            db_queue.put(('mark_batch_processed', batch_id))

            embeddings_writer_logger.info(f"Embeddings for batch {batch_id} saved successfully.")
        except Exception as e:
            session.rollback()
            embeddings_writer_logger.error(f"Error saving embeddings for batch {batch_id}: {e}")
            embeddings_writer_logger.debug(traceback.format_exc())
        finally:
            embeddings_queue.task_done()

        processing_time = time.time() - start_time
        stats_collector.add_batch_processing_time('embeddings_writer', processing_time)
        time.sleep(1)  # Small delay to reduce load
    session.close()

# Archiver thread
def archiver_thread(archive_queue, engine, archive_type, archive_config, stats_collector, log_level, log_output):
    global MACHINE_ID
    # Set up logger for this function
    log_filename = os.path.join('logs', 'archiver', f'archiver_{MACHINE_ID}.log')
    archiver_logger = configure_thread_logging('archiver', log_filename, log_level, log_output)

    SessionFactory = get_session_factory(engine)
    session = SessionFactory()

    while True:
        archive_info = archive_queue.get()
        if archive_info is None:
            archive_queue.task_done()
            break  # Termination signal

        start_time = time.time()
        batch_id = archive_info['batch_id']
        batch_dir = archive_info['batch_dir']
        filenames = archive_info['filenames']
        image_ids = archive_info['image_ids']
        filename_to_id = archive_info['filename_to_id']

        archiver_logger.info(f"Starting archiving for batch {batch_id}")

        # Archive images
        archive_urls = archive_batch(batch_dir, batch_id, archive_type, archive_config, archiver_logger)

        # Now, store the archive_urls in the database
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

        # Remove batch directory
        try:
            shutil.rmtree(batch_dir)
            archiver_logger.info(f"Removed temporary directory for batch {batch_id}")
        except Exception as e:
            archiver_logger.error(f"Error deleting batch directory {batch_dir}: {e}")

        # Update statistics
        stats_collector.increment_batches_archived()

        processing_time = time.time() - start_time
        stats_collector.add_batch_processing_time('archiver', processing_time)

        archive_queue.task_done()
        time.sleep(1)  # Small delay to reduce load
    session.close()

# Functions to archive batch
def archive_batch(batch_dir, batch_id, archive_type, archive_config, logger):
    if archive_type == 's3':
        return archive_batch_to_s3(batch_dir, archive_config, logger)
    elif archive_type == 'azure':
        return archive_batch_to_azure_blob(batch_dir, archive_config, logger)
    elif archive_type == 'ftp':
        return archive_batch_to_ftp(batch_dir, archive_config, logger)
    else:
        logger.error(f"Unknown archive type: {archive_type}")
        return {}

def archive_batch_to_s3(batch_dir, archive_config, logger):
    s3_endpoint = archive_config.get('s3_endpoint')
    s3_access_key = archive_config.get('s3_access_key')
    s3_secret_key = archive_config.get('s3_secret_key')
    s3_bucket = archive_config.get('s3_bucket')

    session = boto3.session.Session()
    s3_client = session.client(
        service_name='s3',
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key
    )

    archive_urls = {}
    for root, dirs, files in os.walk(batch_dir):
        for file in files:
            local_path = os.path.join(root, file)
            s3_key = f"{os.path.basename(batch_dir)}/{file}"
            try:
                s3_client.upload_file(local_path, s3_bucket, s3_key)
                archive_url = f"{s3_endpoint}/{s3_bucket}/{s3_key}"
                archive_urls[file] = archive_url
                logger.info(f"Uploaded {file} to S3 at {archive_url}")
            except Exception as e:
                logger.error(f"Failed to upload {file} to S3: {e}")
    return archive_urls

def archive_batch_to_azure_blob(batch_dir, archive_config, logger):
    connection_string = archive_config.get('azure_connection_string')
    container_name = archive_config.get('azure_container_name')

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    archive_urls = {}
    for root, dirs, files in os.walk(batch_dir):
        for file in files:
            local_path = os.path.join(root, file)
            blob_name = f"{os.path.basename(batch_dir)}/{file}"
            try:
                with open(local_path, "rb") as data:
                    container_client.upload_blob(name=blob_name, data=data)
                archive_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"
                archive_urls[file] = archive_url
                logger.info(f"Uploaded {file} to Azure Blob at {archive_url}")
            except Exception as e:
                logger.error(f"Failed to upload {file} to Azure Blob: {e}")
    return archive_urls

def archive_batch_to_ftp(batch_dir, archive_config, logger):
    ftp_host = archive_config.get('ftp_host')
    ftp_port = archive_config.get('ftp_port', 21)
    ftp_username = archive_config.get('ftp_username')
    ftp_password = archive_config.get('ftp_password')
    ftp_directory = archive_config.get('ftp_directory', '/')

    archive_urls = {}
    try:
        ftp = ftplib.FTP()
        ftp.connect(ftp_host, ftp_port)
        ftp.login(ftp_username, ftp_password)
        ftp.cwd(ftp_directory)

        for root, dirs, files in os.walk(batch_dir):
            for file in files:
                local_path = os.path.join(root, file)
                with open(local_path, 'rb') as f:
                    ftp.storbinary(f'STOR {file}', f)
                archive_url = f"ftp://{ftp_host}{ftp_directory}/{file}"
                archive_urls[file] = archive_url
                logger.info(f"Uploaded {file} to FTP at {archive_url}")
        ftp.quit()
    except Exception as e:
        logger.error(f"Failed to upload files to FTP: {e}")
    return archive_urls

# Main function
def main():
    global MACHINE_ID
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
    parser.add_argument('--archive-type', type=str, choices=['s3', 'azure', 'ftp'], help='Type of archive storage')
    parser.add_argument('--archive-config', type=str, help='Path to archive configuration file')
    parser.add_argument('--archive-threads', type=int, default=int(os.environ.get('ARCHIVE_THREADS', 4)),
                        help='Number of archiver threads (default 4)')

    args = parser.parse_args()

    # Read environment variables
    MACHINE_ID = int(os.environ.get('MACHINE_ID', '0'))
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

    total_pages_to_process = args.limit

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

    print(f"MACHINE_ID: {MACHINE_ID}, TOTAL_MACHINES: {TOTAL_MACHINES}, DOWNLOAD_DIR: {DOWNLOAD_DIR}, "
          f"MAX_BATCHES_ON_DISK: {MAX_BATCHES_ON_DISK}, DOWNLOAD_THREADS: {DOWNLOAD_THREADS}, "
          f"BATCH_SIZE: {BATCH_SIZE}, REPORT_DIR: {REPORT_DIR}, STATS_INTERVAL: {STATS_INTERVAL}, "
          f"LOG_LEVEL: {args.log_level}, LOG_OUTPUT: {LOG_OUTPUT}, LOGGERS: {LOGGERS}, "
          f"ARCHIVE_ENABLED: {archive_enabled}, ARCHIVE_TYPE: {archive_type}, ARCHIVE_THREADS: {ARCHIVE_THREADS}")

    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    # Set up logging for main
    log_filename = os.path.join('logs', 'main', f'main_{MACHINE_ID}.log')
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
    log_file_path = os.path.join('logs', f'images_without_faces_{MACHINE_ID}.log')
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    images_without_faces_log_file = open(log_file_path, 'a')

    # Start threads
    db_writer = threading.Thread(target=db_writer_thread, args=(db_queue, batch_ready_queue, engine, stats_collector, LOG_LEVEL, LOG_OUTPUT))
    downloader = threading.Thread(target=downloader_thread, args=(page_queue, batch_queue, db_queue, batch_ready_queue, DOWNLOAD_DIR, DOWNLOAD_THREADS, stats_collector, LOG_LEVEL, LOG_OUTPUT))
    embeddings_writer = threading.Thread(target=embeddings_writer_thread, args=(embeddings_queue, db_queue, engine, stats_collector, LOG_LEVEL, LOG_OUTPUT))
    processor = threading.Thread(target=processing_thread, args=(batch_queue, embeddings_queue, archive_queue, model, mtcnn, device, engine, BATCH_SIZE, REPORT_DIR, stats_collector, LOG_LEVEL, LOG_OUTPUT, images_without_faces_log_file))
    stats_logger = threading.Thread(target=stats_logger_thread, args=(stats_collector, STATS_INTERVAL, stop_event, LOG_LEVEL, LOG_OUTPUT, total_pages_to_process, page_queue, batch_queue, embeddings_queue, db_queue, archive_queue))

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

    # Fill the page queue for processing
    for page_num in range(args.start, args.start + args.limit):
        if (page_num % TOTAL_MACHINES) != MACHINE_ID:
            continue  # This page is not for this machine

        page_queue.put(page_num)

    # Finish the page queue
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
