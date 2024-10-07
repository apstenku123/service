# file: processor.py
# directory: .
import os
import time
import torch
from utils import configure_thread_logging, get_session_factory, preprocess_image
from models import Batch, Image, BatchImage, BatchLog, HostLog
import socket
import traceback
import numpy as np

# Импортируем необходимые модули для InsightFace
from insightface.app import FaceAnalysis
import cv2

def processing_thread(batch_queue, embeddings_queue, archive_queue, model, mtcnn, device, engine, batch_size, report_dir, stats_collector, log_level, log_output, images_without_faces_log_file):
    global MACHINE_ID
    # Set up logger for this function
    log_filename = f'logs/embedding_processor/embedding_processor_{MACHINE_ID}.log'
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

    # Инициализируем модель InsightFace
    app = FaceAnalysis(allowed_modules=['recognition'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0 if device.type == 'cuda' else -1)

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
            image_ids_list = []
            filenames_list = []
            for img in images:
                local_path = os.path.join(batch_dir, img.filename)
                if not os.path.exists(local_path):
                    embedding_processor_logger.error(f"File not found: {local_path}")
                    continue
                image_paths.append(local_path)
                image_ids_list.append(img.id)
                filenames_list.append(img.filename)

            # Preprocess images (load and convert)
            images_data_list = []
            valid_image_ids = []
            valid_filenames = []
            for idx, path in enumerate(image_paths):
                img_rgb = preprocess_image(path)
                if img_rgb is not None:
                    images_data_list.append(img_rgb)
                    valid_image_ids.append(image_ids_list[idx])
                    valid_filenames.append(filenames_list[idx])
                else:
                    embedding_processor_logger.error(f"Failed to load image: {path}")

            total_images = len(images_data_list)
            total_faces = 0
            images_with_faces = 0
            images_without_faces = 0

            # Process images in batches
            num_batches = (total_images + batch_size - 1) // batch_size
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, total_images)
                batch_images = images_data_list[batch_start:batch_end]
                batch_image_ids = valid_image_ids[batch_start:batch_end]
                batch_filenames = valid_filenames[batch_start:batch_end]

                # Detect faces using MTCNN in batch
                try:
                    boxes_list, _ = mtcnn.detect(batch_images)
                except Exception as e:
                    embedding_processor_logger.error(f"Face detection error in batch {batch_idx}: {e}")
                    continue

                # Collect faces and metadata
                face_imgs = []
                face_image_ids = []
                face_filenames = []
                for idx_in_batch, boxes in enumerate(boxes_list):
                    if boxes is None:
                        images_without_faces += 1
                        stats_collector.increment_images_without_faces()
                        embedding_processor_logger.info(f"No faces detected in image: {batch_filenames[idx_in_batch]}")
                        # Write full URL to log file
                        full_url = filename_to_url[batch_filenames[idx_in_batch]]
                        images_without_faces_log_file.write(f"{full_url}\n")
                        continue

                    images_with_faces += 1
                    stats_collector.increment_images_with_faces()
                    num_faces = boxes.shape[0]
                    total_faces += num_faces
                    stats_collector.increment_faces_found(num_faces)

                    for face_idx, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.astype(int)
                        face_img = batch_images[idx_in_batch][y1:y2, x1:x2]
                        face_imgs.append(face_img)
                        face_image_ids.append(batch_image_ids[idx_in_batch])
                        face_filenames.append(batch_filenames[idx_in_batch])

                if not face_imgs:
                    continue

                # Prepare faces for InceptionResnetV1
                face_tensors = []
                for face_img in face_imgs:
                    face_tensor = preprocess_face(face_img, device)
                    face_tensors.append(face_tensor)
                face_tensors = torch.stack(face_tensors).to(device)

                # Get embeddings from InceptionResnetV1
                with torch.no_grad():
                    embeddings = model(face_tensors).cpu().numpy()

                # Prepare faces for InsightFace
                face_inputs = []
                for face_img in face_imgs:
                    face_input = preprocess_face_insightface(face_img)
                    face_inputs.append(face_input)
                face_inputs = np.vstack(face_inputs)

                # Get embeddings from InsightFace in batch
                recognition_model = app.models.get('recognition', None)
                if recognition_model is None:
                    insightface_embeddings = [None] * len(face_imgs)
                else:
                    insightface_embeddings = recognition_model(face_inputs).tolist()

                # Collect embeddings data
                for idx_face in range(len(face_imgs)):
                    embedding = embeddings[idx_face].tolist()
                    insightface_embedding = insightface_embeddings[idx_face]
                    embeddings_data.append({
                        'image_id': face_image_ids[idx_face],
                        'filename': face_filenames[idx_face],
                        'embedding': embedding,
                        'insightface_embedding': insightface_embedding
                    })

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

def preprocess_face(face_img, device):
    from PIL import Image
    from torchvision import transforms

    face_img = Image.fromarray(face_img)
    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    face_tensor = preprocess(face_img).to(device)
    return face_tensor

def preprocess_face_insightface(face_img):
    # Преобразуем изображение в BGR, если оно в RGB
    if face_img.shape[2] == 3:
        face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
    else:
        face_img_bgr = face_img

    # Предобрабатываем лицо для InsightFace
    face_img_resized = cv2.resize(face_img_bgr, (112, 112))
    face_img_normalized = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2RGB)
    face_img_normalized = (face_img_normalized / 127.5) - 1.0  # Нормализация от -1 до 1
    face_img_transposed = np.transpose(face_img_normalized, (2, 0, 1))  # HWC to CHW
    face_img_input = np.expand_dims(face_img_transposed, axis=0).astype(np.float32)
    return face_img_input
