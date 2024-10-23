# file: test444.py
# directory: test_nv
import argparse
import os
import sys
import time
import logging
import torch
import cv2
import cupy as cp
from PIL import Image

from facenet_pytorch import MTCNN, InceptionResnetV1
from insightface.app import FaceAnalysis
from insightface.utils import face_align


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face Embedding Tester')
    parser.add_argument('-s', '--image_file', type=str, help='Path to the image file')
    parser.add_argument('-f', '--directory', type=str, help='Path to the directory with images')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug messages')
    parser.add_argument('-l', '--log', action='store_true', help='Write messages to log file')
    args = parser.parse_args()

    if not args.image_file and not args.directory:
        print("Please provide an image file with -s or a directory with -f.")
        sys.exit(1)

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = logging.getLogger('FaceEmbeddingTester')
    logger.setLevel(log_level)

    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # If log to file is enabled, add file handler
    if args.log:
        pid = os.getpid()
        log_filename = f'messages_{pid}.log'
        fh = logging.FileHandler(log_filename)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Initialize device (CUDA if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

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

    # Store the start time after models are loaded
    processing_start_time = time.time()

    # Initialize statistics counters
    images_processed = 0
    total_images = 0
    total_faces_mtcnn = 0
    total_embeddings_facenet = 0
    total_embeddings_insightface = 0

    # Function to process a single image
    def process_image(image_path):
        nonlocal images_processed, total_images, total_faces_mtcnn, total_embeddings_facenet, total_embeddings_insightface

        # Read the original image using OpenCV
        img_bgr_orig = cv2.imread(image_path)
        if img_bgr_orig is None:
            logger.error(f"Failed to load image: {image_path}")
            return

        # Convert the original image to RGB
        img_rgb_orig = cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2RGB)

        # Detect faces with MTCNN on original image
        boxes, probs, landmarks = mtcnn.detect(img_rgb_orig, landmarks=True)

        images_processed += 1
        total_images += 1

        if boxes is not None and landmarks is not None:
            num_faces = boxes.shape[0]
            total_faces_mtcnn += num_faces
            logger.debug(f'Image: {os.path.basename(image_path)} - Number of faces detected by MTCNN: {num_faces}')

            # Convert boxes to integers
            boxes = boxes.astype(int)

            # Get aligned faces for InceptionResnetV1 from original image
            aligned_faces = mtcnn.extract(img_rgb_orig, boxes, save_path=None)

            if aligned_faces is not None and len(aligned_faces) > 0:
                # Use aligned_faces directly without stacking
                face_tensors = aligned_faces.pin_memory().to(device, non_blocking=True)
                # Get embeddings from InceptionResnetV1
                with torch.no_grad():
                    embeddings_facenet = facenet_model(face_tensors).cpu().numpy()
                num_embeddings_facenet = embeddings_facenet.shape[0]
                total_embeddings_facenet += num_embeddings_facenet
                logger.debug(f'Number of embeddings from InceptionResnetV1: {num_embeddings_facenet}')
            else:
                logger.error("Failed to get aligned faces for InceptionResnetV1.")
                return

            # Get embeddings from InsightFace
            embeddings_insightface = []
            for i in range(num_faces):
                # landmarks[i] has shape (5, 2)
                landmark = landmarks[i]
                # Process landmark with CuPy
                landmark_cp = cp.asarray(landmark, dtype=cp.float32)
                if landmark_cp.shape != (5, 2):
                    logger.error(f"Invalid landmark shape for face {i}: {landmark_cp.shape}. Skipping.")
                    continue

                if cp.isnan(landmark_cp).any() or cp.isinf(landmark_cp).any():
                    logger.error(f"Invalid landmark values for face {i}. Skipping.")
                    continue

                # Convert landmark back to host memory for face_align
                landmark_np = cp.asnumpy(landmark_cp)

                # Use face_align.norm_crop to align the face on original image
                face_aligned = face_align.norm_crop(img_rgb_orig, landmark=landmark_np, image_size=112)

                # Convert the image to BGR
                face_aligned_bgr = cv2.cvtColor(face_aligned, cv2.COLOR_RGB2BGR)

                # Ensure the image is of type uint8
                face_aligned_bgr_cp = cp.asarray(face_aligned_bgr.astype('uint8'))

                # Get embedding with recognizer
                embedding_insight = recognizer.get_feat([cp.asnumpy(face_aligned_bgr_cp)])
                embeddings_insightface.append(embedding_insight)
                total_embeddings_insightface += 1

            logger.debug(f'Number of embeddings from InsightFace: {len(embeddings_insightface)}')

        else:
            logger.debug(f'MTCNN did not detect faces in image: {os.path.basename(image_path)}')

        # Clear GPU cache every 400 images
        if images_processed % 400 == 0:
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared.")

    # Function to process images in batch
    def process_images_batch(image_paths):
        nonlocal images_processed, total_images, total_faces_mtcnn, total_embeddings_facenet, total_embeddings_insightface

        # Read and preprocess all images
        images_orig = []
        valid_image_paths = []
        for image_path in image_paths:
            img_bgr_orig = cv2.imread(image_path)
            if img_bgr_orig is None:
                logger.error(f"Failed to load image: {image_path}")
                continue

            # Convert original image to RGB
            img_rgb_orig = cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2RGB)

            images_orig.append(img_rgb_orig)
            valid_image_paths.append(image_path)

        if not images_orig:
            logger.error("No valid images to process.")
            return

        # Detect faces with MTCNN in batch on original images
        boxes_batch, probs_batch, landmarks_batch = mtcnn.detect(images_orig, landmarks=True)

        for idx, image_path in enumerate(valid_image_paths):
            img_rgb_orig = images_orig[idx]

            boxes = boxes_batch[idx]
            landmarks = landmarks_batch[idx]

            images_processed += 1
            total_images += 1

            if boxes is not None and landmarks is not None:
                num_faces = boxes.shape[0]
                total_faces_mtcnn += num_faces
                logger.debug(f'Image: {os.path.basename(image_path)} - Number of faces detected by MTCNN: {num_faces}')

                # Convert boxes to integers
                boxes = boxes.astype(int)

                # Get aligned faces for InceptionResnetV1 from original images
                aligned_faces = mtcnn.extract(img_rgb_orig, boxes, save_path=None)

                if aligned_faces is not None and len(aligned_faces) > 0:
                    # Use aligned_faces directly without stacking
                    face_tensors = aligned_faces.pin_memory().to(device, non_blocking=True)
                    # Get embeddings from InceptionResnetV1
                    with torch.no_grad():
                        embeddings_facenet = facenet_model(face_tensors).cpu().numpy()
                    num_embeddings_facenet = embeddings_facenet.shape[0]
                    total_embeddings_facenet += num_embeddings_facenet
                    logger.debug(f'Number of embeddings from InceptionResnetV1: {num_embeddings_facenet}')
                else:
                    logger.error("Failed to get aligned faces for InceptionResnetV1.")
                    continue

                # Get embeddings from InsightFace
                embeddings_insightface = []
                for i in range(num_faces):
                    # landmarks[i] has shape (5, 2)
                    landmark = landmarks[i]
                    # Process landmark with CuPy
                    landmark_cp = cp.asarray(landmark, dtype=cp.float32)
                    if landmark_cp.shape != (5, 2):
                        logger.error(f"Invalid landmark shape for face {i}: {landmark_cp.shape}. Skipping.")
                        continue

                    if cp.isnan(landmark_cp).any() or cp.isinf(landmark_cp).any():
                        logger.error(f"Invalid landmark values for face {i}. Skipping.")
                        continue

                    # Convert landmark back to host memory for face_align
                    landmark_np = cp.asnumpy(landmark_cp)

                    # Use face_align.norm_crop to align the face on original image
                    face_aligned = face_align.norm_crop(img_rgb_orig, landmark=landmark_np, image_size=112)

                    # Convert the image to BGR
                    face_aligned_bgr = cv2.cvtColor(face_aligned, cv2.COLOR_RGB2BGR)

                    # Ensure the image is of type uint8
                    face_aligned_bgr_cp = cp.asarray(face_aligned_bgr.astype('uint8'))

                    # Get embedding with recognizer
                    embedding_insight = recognizer.get_feat([cp.asnumpy(face_aligned_bgr_cp)])
                    embeddings_insightface.append(embedding_insight)
                    total_embeddings_insightface += 1

                logger.debug(f'Number of embeddings from InsightFace: {len(embeddings_insightface)}')

            else:
                logger.debug(f'MTCNN did not detect faces in image: {os.path.basename(image_path)}')

            # Clear GPU cache every 400 images
            if images_processed % 400 == 0:
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared.")

    if args.image_file:
        if not os.path.exists(args.image_file):
            logger.error(f"Image file {args.image_file} does not exist.")
            sys.exit(1)
        process_image(args.image_file)

    elif args.directory:
        if not os.path.exists(args.directory):
            logger.error(f"Directory {args.directory} does not exist.")
            sys.exit(1)

        # Get list of image files in the directory
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [os.path.join(args.directory, f) for f in os.listdir(args.directory)
                       if f.lower().endswith(supported_formats)]

        if not image_files:
            logger.error(f"No image files found in directory {args.directory}.")
            sys.exit(1)

        # Read image sizes and group them
        image_sizes = {}
        for image_file in image_files:
            try:
                with Image.open(image_file) as img:
                    size = img.size  # (width, height)
                    if size in image_sizes:
                        image_sizes[size].append(image_file)
                    else:
                        image_sizes[size] = [image_file]
            except Exception as e:
                logger.error(f"Error reading image {image_file}: {e}")

        total_files = len(image_files)
        num_grouped_images = sum(len(files) for size, files in image_sizes.items() if len(files) > 1)
        num_ungrouped_images = sum(len(files) for size, files in image_sizes.items() if len(files) == 1)
        num_groups = len([size for size, files in image_sizes.items() if len(files) > 1])

        logger.info(f"Total images: {total_files}")
        logger.info(f"Number of grouped images: {num_grouped_images} in {num_groups} groups")
        logger.info(f"Number of ungrouped images: {num_ungrouped_images}")

        # Process images grouped by size
        for size, files in image_sizes.items():
            num_files = len(files)
            if num_files == 0:
                continue
            else:
                # Process images in batches of up to 16 images
                for i in range(0, num_files, 16):
                    batch_files = files[i:i+16]
                    if len(batch_files) == 1:
                        process_image(batch_files[0])
                    else:
                        process_images_batch(batch_files)

    # Calculate processing time
    processing_end_time = time.time()
    total_processing_time = processing_end_time - processing_start_time

    # Calculate images per second
    if total_processing_time > 0:
        images_per_second = total_images / total_processing_time
        faces_per_second = total_faces_mtcnn / total_processing_time
    else:
        images_per_second = 0
        faces_per_second = 0

    # Output statistics
    logger.info("Processing completed.")
    logger.info(f"Total processing time (excluding model loading): {total_processing_time:.2f} seconds")
    logger.info(f"Total images processed: {total_images}")
    logger.info(f"Images per second: {images_per_second:.2f}")
    logger.info(f"Total faces detected by MTCNN: {total_faces_mtcnn}")
    logger.info(f"Faces per second: {faces_per_second:.2f}")
    logger.info(f"Total embeddings from InceptionResnetV1: {total_embeddings_facenet}")
    logger.info(f"Total embeddings from InsightFace: {total_embeddings_insightface}")


if __name__ == '__main__':
    main()
