# file: archiver.py
# directory: .

import threading
import time
import os
import shutil
import traceback
import socket

from utils import configure_thread_logging, get_session_factory
from models import ArchivedImage, HostLog
import config  # Импортируем модуль конфигурации

# Импортируем функции архивации из оригинального archiver.py
from archiver_functions import (
    archive_batch_to_s3,
    archive_batch_to_azure_blob,
    archive_batch_to_ftp,
    archive_batch_to_sftp,
)


def archiver_thread(archive_queue, engine, archive_type, archive_config, stats_collector, log_level, log_output):
    # Set up logger for this function
    log_filename = f'logs/archiver/archiver_{config.MACHINE_ID}.log'
    archiver_logger = configure_thread_logging('archiver', log_filename, log_level, log_output)

    SessionFactory = get_session_factory(engine)
    session = SessionFactory()
    host_name = socket.gethostname()

    # Check if HostLog exists
    existing_host_log = session.query(HostLog).filter_by(host_name=host_name, function_name='archiver', log_file=log_filename).first()
    if not existing_host_log:
        host_log = HostLog(host_name=host_name, function_name='archiver', log_file=log_filename)
        session.add(host_log)
        session.commit()
    else:
        host_log = existing_host_log

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
        filename_to_id = dict(zip(filenames, image_ids))

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


def archive_batch(batch_dir, batch_id, archive_type, archive_config, logger):
    if archive_type == 's3':
        return archive_batch_to_s3(batch_dir, archive_config, logger)
    elif archive_type == 'azure':
        return archive_batch_to_azure_blob(batch_dir, archive_config, logger)
    elif archive_type == 'ftp':
        return archive_batch_to_ftp(batch_dir, archive_config, logger)
    elif archive_type == 'sftp':
        return archive_batch_to_sftp(batch_dir, archive_config, logger)
    else:
        logger.error(f"Unknown archive type: {archive_type}")
        return {}
