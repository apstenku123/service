# file: embeddings_writer.py
# directory: .

import time
import traceback
import socket

from utils import configure_thread_logging, get_session_factory
from models import ImageEmbedding, HostLog
import config  # Импортируем модуль конфигурации


def embeddings_writer_thread(embeddings_queue, db_queue, engine, stats_collector, log_level, log_output):
    # Set up logger for this function
    log_filename = f'logs/embeddings_writer/embeddings_writer_{config.MACHINE_ID}.log'
    embeddings_writer_logger = configure_thread_logging('embeddings_writer', log_filename, log_level, log_output)

    SessionFactory = get_session_factory(engine)
    session = SessionFactory()
    host_name = socket.gethostname()

    # Check if HostLog exists
    existing_host_log = session.query(HostLog).filter_by(host_name=host_name, function_name='embeddings_writer', log_file=log_filename).first()
    if not existing_host_log:
        host_log = HostLog(host_name=host_name, function_name='embeddings_writer', log_file=log_filename)
        session.add(host_log)
        session.commit()
    else:
        host_log = existing_host_log

    while True:
        embeddings_info = embeddings_queue.get()
        embeddings_writer_logger.info("Received embeddings_info from queue.")
        if embeddings_info is None:
            embeddings_writer_logger.info("Termination signal received. Exiting embeddings_writer_thread.")
            embeddings_queue.task_done()
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
                embedding_vector = data.get('embedding')
                insightface_embedding_vector = data.get('insightface_embedding')

                if embedding_vector is None and insightface_embedding_vector is None:
                    embeddings_writer_logger.error(f"No embeddings for image_id {data['image_id']}")
                    continue

                embedding = ImageEmbedding(
                    image_id=data['image_id'],
                    filename=data['filename'],
                    embedding=embedding_vector,
                    insightface_embedding=insightface_embedding_vector
                )
                embeddings_objects.append(embedding)
            session.bulk_save_objects(embeddings_objects)
            session.commit()
            embeddings_writer_logger.info(f"Embeddings for batch {batch_id} committed to database.")

            # Update statistics
            embeddings_count = len(embeddings_objects)
            stats_collector.increment_embeddings_uploaded(embeddings_count, embedding_type='both')

            # Mark batch as processed
            db_queue.put(('mark_batch_processed', batch_id))

            embeddings_writer_logger.info(f"Embeddings for batch {batch_id} saved successfully.")
        except Exception as e:
            session.rollback()
            embeddings_writer_logger.error(f"Error saving embeddings for batch {batch_id}: {e}", exc_info=True)
        finally:
            embeddings_queue.task_done()

        processing_time = time.time() - start_time
        stats_collector.add_batch_processing_time('embeddings_writer', processing_time)
        time.sleep(1)  # Small delay to reduce load
    session.close()
