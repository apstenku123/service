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
from models import Batch, Image, BatchImage, BatchLog, HostLog, ImageEmbedding
from facenet_pytorch import MTCNN, InceptionResnetV1
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import config  # Import the configuration module


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

    # Initialize MTCNN with keep_all=True
    mtcnn = MTCNN(keep_all=True, device=device)

    # Initialize InceptionResnetV1 model
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Initialize InsightFace model
    app = FaceAnalysis(allowed_modules=['detection', 'recognition'],
                       providers=['CUDAExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider'])
    app.prepare(ctx_id=0 if device.type == 'cuda' else -1, det_size=(640, 640))

    # Get the recognition model
    recognizer = app.models['recognition']

    # Target size for resizing images for MTCNN
    mtcnn_target_size = (1280, 960)  # Width x Height

    # MTCNN processing batch size
    mtcnn_batch_size = 32  # Adjust based on your GPU memory

    # Total images processed across all batches
    total_images_processed = 0

    # Function to resize image for MTCNN and compute scaling factors
    def resize_image_for_mtcnn(image, target_size):
        original_size = image.shape[:2]  # (height, width)
        h, w = original_size
        target_w, target_h = target_size

        # Calculate scaling factors
        scale_w = target_w / w
        scale_h = target_h / h

        # Resize image
        resized_image = cv2.resize(image, (target_w, target_h))

        return resized_image, scale_w, scale_h

    # Function to adjust boxes and landmarks back to original image coordinates
    def adjust_boxes_landmarks(boxes, landmarks, scale_w, scale_h):
        # Adjust boxes
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / scale_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / scale_h

        # Adjust landmarks
        landmarks[:, :, 0] = landmarks[:, :, 0] / scale_w
        landmarks[:, :, 1] = landmarks[:, :, 1] / scale_h

        return boxes, landmarks

    # Function to process images in batch
    def process_images_batch(image_paths, valid_image_ids, valid_filenames, valid_image_urls, logger, mtcnn_batch_size):
        embeddings_data = []
        total_images_in_batch = len(image_paths)
        total_faces = 0
        images_with_faces = 0
        images_without_faces = 0

        nonlocal total_images_processed  # To update the outer variable

        # Process images in sub-batches for MTCNN
        for batch_start in range(0, len(image_paths), mtcnn_batch_size):
            batch_end = batch_start + mtcnn_batch_size
            batch_image_paths = image_paths[batch_start:batch_end]
            batch_image_ids = valid_image_ids[batch_start:batch_end]
            batch_filenames = valid_filenames[batch_start:batch_end]
            batch_image_urls = valid_image_urls[batch_start:batch_end]

            # Read and preprocess images in the sub-batch
            images_orig = []
            images_resized = []
            scales_w = []
            scales_h = []
            valid_indices = []

            for idx, image_path in enumerate(batch_image_paths):
                img_bgr_orig = cv2.imread(image_path)
                if img_bgr_orig is None:
                    logger.error(f"Failed to load image: {image_path}")
                    continue

                # Convert original image to RGB
                img_rgb_orig = cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2RGB)

                # Resize image for MTCNN
                img_rgb_resized, scale_w, scale_h = resize_image_for_mtcnn(img_rgb_orig, mtcnn_target_size)

                images_orig.append(img_rgb_orig)
                images_resized.append(img_rgb_resized)
                scales_w.append(scale_w)
                scales_h.append(scale_h)
                valid_indices.append(idx)

            if not images_resized:
                logger.error("No valid images to process in sub-batch.")
                continue

            # Detect faces with MTCNN in batch on resized images
            boxes_batch_resized, probs_batch_resized, landmarks_batch_resized = mtcnn.detect(images_resized, landmarks=True)

            for idx_in_subbatch, idx in enumerate(valid_indices):
                img_rgb_orig = images_orig[idx_in_subbatch]
                scale_w = scales_w[idx_in_subbatch]
                scale_h = scales_h[idx_in_subbatch]
                image_id = batch_image_ids[idx]
                filename = batch_filenames[idx]
                image_url = batch_image_urls[idx]

                boxes_resized = boxes_batch_resized[idx_in_subbatch]
                landmarks_resized = landmarks_batch_resized[idx_in_subbatch]

                total_images_processed += 1

                if boxes_resized is not None and landmarks_resized is not None:
                    num_faces = boxes_resized.shape[0]
                    total_faces += num_faces
                    images_with_faces += 1
                    stats_collector.increment_faces_found(num_faces)
                    stats_collector.increment_images_with_faces()

                    # Adjust boxes and landmarks back to original image coordinates
                    boxes_orig, landmarks_orig = adjust_boxes_landmarks(boxes_resized, landmarks_resized, scale_w, scale_h)

                    # Convert boxes to integers
                    boxes_orig = boxes_orig.astype(int)

                    # Get aligned faces for InceptionResnetV1 from original images
                    aligned_faces = mtcnn.extract(img_rgb_orig, boxes_orig, save_path=None)

                    if aligned_faces is not None and len(aligned_faces) > 0:
                        # Use aligned_faces directly without stacking
                        face_tensors = aligned_faces.pin_memory().to(device, non_blocking=True)
                        # Get embeddings from InceptionResnetV1
                        with torch.no_grad():
                            embeddings_facenet = facenet_model(face_tensors).cpu().numpy()
                        num_embeddings_facenet = embeddings_facenet.shape[0]
                        logger.debug(f'Number of embeddings from InceptionResnetV1: {num_embeddings_facenet}')

                        # Get embeddings from InsightFace
                        for i in range(num_faces):
                            # landmarks_orig[i] has shape (5, 2)
                            landmark = landmarks_orig[i]

                            # Use face_align.norm_crop to align the face on original image
                            face_aligned = face_align.norm_crop(img_rgb_orig, landmark=landmark, image_size=112)

                            # Convert the image to BGR
                            face_aligned_bgr = cv2.cvtColor(face_aligned, cv2.COLOR_RGB2BGR)

                            # Ensure the image is of type uint8
                            face_aligned_bgr = face_aligned_bgr.astype('uint8')

                            # Get embedding with recognizer
                            embedding_insight = recognizer.get_feat(face_aligned_bgr)

                            # Collect embeddings_data
                            embeddings_data.append({
                                'image_id': image_id,
                                'filename': filename,
                                'embedding': embeddings_facenet[i].tolist(),
                                'insightface_embedding': embedding_insight.tolist()
                            })
                    else:
                        logger.error("Failed to get aligned faces for InceptionResnetV1.")
                        images_without_faces += 1
                        stats_collector.increment_images_without_faces()
                        logger.info(f"No faces detected in image: {filename}")

                        # Write image URL to log file
                        images_without_faces_log_file.write(f"{image_url}\n")
                else:
                    logger.debug(f'MTCNN did not detect faces in image: {filename}')
                    images_without_faces += 1
                    stats_collector.increment_images_without_faces()
                    logger.info(f"No faces detected in image: {filename}")

                    # Write image URL to log file
                    images_without_faces_log_file.write(f"{image_url}\n")

                # Clear GPU cache every 400 images
                if total_images_processed % 400 == 0:
                    torch.cuda.empty_cache()
                    logger.info("GPU cache cleared.")

        return embeddings_data, total_images_in_batch, total_faces, images_with_faces, images_without_faces

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

        try:
            image_paths = [os.path.join(batch_dir, filename) for filename in filenames]
            valid_image_ids = []
            valid_filenames = []
            valid_image_urls = []
            valid_image_paths = []

            # Collect valid image data
            for idx, path in enumerate(image_paths):
                if os.path.isfile(path):
                    valid_image_paths.append(path)
                    valid_image_ids.append(filename_to_id[filenames[idx]])
                    valid_filenames.append(filenames[idx])
                    valid_image_urls.append(filename_to_url[filenames[idx]])
                else:
                    embedding_processor_logger.error(f"Image file does not exist: {path}")

            # Now process images in batch
            (embeddings_data, total_images_processed_in_batch, total_faces_detected,
                images_with_faces_count, images_without_faces_count) = process_images_batch(
                     valid_image_paths, valid_image_ids, valid_filenames, valid_image_urls,
                     embedding_processor_logger, mtcnn_batch_size)

            total_images = total_images_processed_in_batch
            total_faces = total_faces_detected
            images_with_faces = images_with_faces_count
            images_without_faces = images_without_faces_count

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
