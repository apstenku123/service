# file: processor.py
# directory: .

import os
import time
import torch
import cv2
import shutil
import socket
import traceback
import numpy as np

from utils import configure_thread_logging, get_session_factory
from models import Batch, Image, BatchImage, BatchLog, HostLog
from insightface.app import FaceAnalysis
import config  # Импортируем модуль конфигурации


def processing_thread(batch_queue, embeddings_queue, archive_queue, device, engine, batch_size, report_dir, stats_collector, log_level, log_output, images_without_faces_log_file, condition):
    # Set up logger for this function
    log_filename = f'logs/embedding_processor/embedding_processor_{config.MACHINE_ID}.log'
    embedding_processor_logger = configure_thread_logging('embedding_processor', log_filename, log_level, log_output)

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

    # Инициализируем модель FaceAnalysis из InsightFace
    app = FaceAnalysis(providers=['CUDAExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider'])
    app.prepare(ctx_id=0 if device.type == 'cuda' else -1)
    # app.max_num_faces = 4  # Опционально ограничить количество лиц

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
        filename_to_id = dict(zip(filenames, image_ids))
        filename_to_url = dict(zip(filenames, image_urls))

        # Check if batch is already processed
        batch = session.query(Batch).filter_by(id=batch_id).first()
        if batch.processed:
            embedding_processor_logger.info(f"Batch {batch_id} is already processed. Skipping.")
            batch_queue.task_done()
            continue

        embedding_processor_logger.info(f"Starting processing of batch {batch_id}")
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

                    # Write image URL to log file
                    images_without_faces_log_file.write(f"{valid_image_urls[idx_in_batch]}\n")

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

            # Mark batch as processed
            batch.processed = True
            session.commit()

            # Update statistics
            stats_collector.increment_images_processed(len(valid_image_ids))
            stats_collector.increment_batches_processed()

            # Associate batch with log file
            batch_log = BatchLog(batch_id=batch_id, host_log_id=host_log.id)
            session.add(batch_log)
            session.commit()

            # Remove batch directory
            shutil.rmtree(batch_dir)
            embedding_processor_logger.info(f"Removed temporary directory for batch {batch_id}")

            # Decrement the counter and notify downloader
            with condition:
                config.current_batches_on_disk -= 1
                condition.notify()

            batch_queue.task_done()
        except Exception as e:
            session.rollback()
            embedding_processor_logger.error(f"Error processing batch {batch_id}: {e}")
            embedding_processor_logger.debug(traceback.format_exc())
            batch_queue.task_done()

        stats_collector.increment_batches_processed_by_processor()
        processing_time = time.time() - start_time
        stats_collector.add_batch_processing_time('embedding_processor', processing_time)
        time.sleep(1)  # Small delay to reduce load
    session.close()
