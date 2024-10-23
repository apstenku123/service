# file: stats_collector.py
# directory: .
import json
import logging
import os
import threading
import time
from logging.handlers import RotatingFileHandler
import config  # Импортируем модуль конфигурации


class StatsCollector:
    def __init__(self):
        self.lock = threading.Lock()
        # Resettable stats
        self.files_downloaded = 0
        self.faces_found = 0
        self.embeddings_uploaded = 0
        self.embeddings_uploaded_by_type = {'embedding': 0, 'insightface_embedding': 0, 'both': 0}
        self.batch_processing_times = {'downloader': [], 'embedding_processor': [], 'embeddings_writer': [], 'archiver': []}
        self.last_reset_time = time.time()
        # Global stats
        self.total_files_downloaded = 0
        self.total_faces_found = 0
        self.total_embeddings_uploaded = 0
        self.total_images_processed = 0
        self.total_batches_processed = 0
        self.images_with_faces = 0
        self.images_without_faces = 0
        self.total_batches_archived = 0
        self.batches_processed_by_processor = 0
        self.start_time = time.time()

    def increment_embeddings_uploaded(self, count=1, embedding_type='both'):
        with self.lock:
            self.embeddings_uploaded += count
            self.total_embeddings_uploaded += count
            self.embeddings_uploaded_by_type[embedding_type] += count

    def increment_batches_processed_by_processor(self, count=1):
        with self.lock:
            self.batches_processed_by_processor += count

    def increment_files_downloaded(self, count=1):
        with self.lock:
            self.files_downloaded += count
            self.total_files_downloaded += count

    def increment_faces_found(self, count=1):
        with self.lock:
            self.faces_found += count
            self.total_faces_found += count

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

    def increment_batches_archived(self, count=1):
        with self.lock:
            self.total_batches_archived += count

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
                'embeddings_uploaded_by_type': self.embeddings_uploaded_by_type.copy(),
                'batch_processing_times': self.batch_processing_times.copy(),
                'elapsed_time': time.time() - self.last_reset_time,
                # Global stats
                'batches_processed_by_processor': self.batches_processed_by_processor,
                'total_files_downloaded': self.total_files_downloaded,
                'total_faces_found': self.total_faces_found,
                'total_embeddings_uploaded': self.total_embeddings_uploaded,
                'total_images_processed': self.total_images_processed,
                'total_batches_processed': self.total_batches_processed,
                'images_with_faces': self.images_with_faces,
                'images_without_faces': self.images_without_faces,
                'total_batches_archived': self.total_batches_archived,
                'start_time': self.start_time,
            }
            # Reset interval stats
            self.files_downloaded = 0
            self.faces_found = 0
            self.embeddings_uploaded = 0
            self.embeddings_uploaded_by_type = {'embedding': 0, 'insightface_embedding': 0, 'both': 0}
            self.batch_processing_times = {key: [] for key in self.batch_processing_times}
            self.last_reset_time = time.time()
        return stats

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

# Statistics logging thread
def stats_logger_thread(stats_collector, interval, stop_event, log_level, log_output, total_pages_to_process, page_queue, batch_queue, embeddings_queue, db_queue):
    # global MACHINE_ID
    stats_logger = configure_thread_logging('stats_logger', f'logs/stats_logger/stats_logger_{config.MACHINE_ID}.log', log_level, log_output)

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
            f"  Embeddings Uploaded by Type:\n"
            f"    embedding: {stats['embeddings_uploaded_by_type']['embedding']}\n"
            f"    insightface_embedding: {stats['embeddings_uploaded_by_type']['insightface_embedding']}\n"
            f"    both: {stats['embeddings_uploaded_by_type']['both']}\n"            
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
        # Записываем статистику в файл
        with open('stats.json', 'w') as f:
            json.dump(stats, f)