# file: test66.py
# directory: test_nv
import argparse
import os
import sys
import time
import logging
import torch
import cv2
import numpy as np

import cupy as cp

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
    parser.add_argument('-b', '--batch', action='store_true', help='Use batch processing with MTCNN')
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

    # Target size for resizing images for MTCNN
    mtcnn_target_size = (1024, 768)  # Width x Height

    # Function to resize images for MTCNN using CUDA in batch
    def batch_resize_images(images, target_size):
        target_w, target_h = target_size
        batch_size = len(images)
        img_height, img_width, channels = images[0].shape

        # Create batch on GPU using CuPy
        batch_gpu = cp.zeros((batch_size, target_h, target_w, channels), dtype=cp.uint8)

        for i, image in enumerate(images):
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)
            gpu_resized = cv2.cuda.resize(gpu_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            batch_gpu[i] = cp.asarray(gpu_resized.download())

        return batch_gpu

    # Function to convert batch of images from BGR to RGB using CUDA
    def batch_convert_bgr_to_rgb(images_gpu):
        return images_gpu[:, :, :, ::-1]  # Simply reverse the color channels for BGR to RGB

    # Function to adjust boxes and landmarks back to original image coordinates
    def adjust_boxes_landmarks(boxes, landmarks, scale_w, scale_h):
        # Adjust boxes
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / scale_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / scale_h

        # Adjust landmarks
        landmarks[:, :, 0] = landmarks[:, :, 0] / scale_w
        landmarks[:, :, 1] = landmarks[:, :, 1] / scale_h

        return boxes, landmarks

    # Теперь функция `adjust_boxes_landmarks` определена выше вызовов `process_image` и `process_images_batch`

    # Function to process images in batch
    def process_images_batch(image_paths):
        nonlocal images_processed, total_images, total_faces_mtcnn, total_embeddings_facenet, total_embeddings_insightface

        # Read and preprocess all images
        images_orig = []
        for image_path in image_paths:
            img_bgr_orig = cv2.imread(image_path)
            if img_bgr_orig is None:
                logger.error(f"Failed to load image: {image_path}")
                continue

            images_orig.append(img_bgr_orig)

        if not images_orig:
            logger.error("No valid images to process.")
            return

        # Convert images to RGB in batch using GPU
        batch_rgb_gpu = batch_convert_bgr_to_rgb(batch_resize_images(images_orig, mtcnn_target_size))

        # Detect faces with MTCNN in batch on resized images
        batch_rgb_cpu = [cp.asnumpy(batch_rgb_gpu[i]) for i in range(len(images_orig))]
        boxes_batch_resized, probs_batch_resized, landmarks_batch_resized = mtcnn.detect(batch_rgb_cpu, landmarks=True)

        for idx, image_path in enumerate(image_paths):
            img_rgb_orig = images_orig[idx]
            img_rgb_resized = batch_rgb_cpu[idx]

            boxes_resized = boxes_batch_resized[idx]
            landmarks_resized = landmarks_batch_resized[idx]

            images_processed += 1
            total_images += 1

            if boxes_resized is not None and landmarks_resized is not None:
                num_faces = boxes_resized.shape[0]
                total_faces_mtcnn += num_faces
                logger.debug(f'Image: {os.path.basename(image_path)} - Number of faces detected by MTCNN: {num_faces}')

                # Adjust boxes and landmarks back to original image coordinates
                scale_w = mtcnn_target_size[0] / img_rgb_orig.shape[1]
                scale_h = mtcnn_target_size[1] / img_rgb_orig.shape[0]
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
                    total_embeddings_facenet += num_embeddings_facenet
                    logger.debug(f'Number of embeddings from InceptionResnetV1: {num_embeddings_facenet}')
                else:
                    logger.error("Failed to get aligned faces for InceptionResnetV1.")
                    continue

                # Get embeddings from InsightFace
                embeddings_insightface = []
                for i in range(num_faces):
                    # landmarks_orig[i] has shape (5, 2)
                    landmark = landmarks_orig[i]
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

                    # Convert the image to RGB
                    face_aligned_rgb = convert_bgr_to_rgb(face_aligned)

                    # Ensure the image is of type uint8
                    face_aligned_rgb_cp = cp.asarray(face_aligned_rgb.astype('uint8'))

                    # Get embedding with recognizer
                    embedding_insight = recognizer.get_feat([cp.asnumpy(face_aligned_rgb_cp)])
                    embeddings_insightface.append(embedding_insight)
                    total_embeddings_insightface += 1

                logger.debug(f'Number of embeddings from InsightFace: {len(embeddings_insightface)}')

            else:
                logger.debug(f'MTCNN did not detect faces in image: {os.path.basename(image_path)}')

            # Clear GPU cache every 400 images
            if images_processed % 400 == 0:
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared.")

    # Function to process a single image
    def process_image(image_path):
        nonlocal images_processed, total_images, total_faces_mtcnn, total_embeddings_facenet, total_embeddings_insightface

        # Read the original image using OpenCV
        img_bgr_orig = cv2.imread(image_path)
        if img_bgr_orig is None:
            logger.error(f"Failed to load image: {image_path}")
            return

        # Resize image for MTCNN using CUDA if available
        img_rgb_resized, scale_w, scale_h = resize_image_for_mtcnn(img_bgr_orig, mtcnn_target_size)

        # Detect faces with MTCNN on resized image
        boxes_resized, probs_resized, landmarks_resized = mtcnn.detect([img_rgb_resized], landmarks=True)

        images_processed += 1
        total_images += 1

        if boxes_resized is not None and landmarks_resized is not None:
            num_faces = boxes_resized.shape[0]
            total_faces_mtcnn += num_faces
            logger.debug(f'Image: {os.path.basename(image_path)} - Number of faces detected by MTCNN: {num_faces}')

            # Adjust boxes and landmarks back to original image coordinates
            boxes_orig, landmarks_orig = adjust_boxes_landmarks(boxes_resized, landmarks_resized, scale_w, scale_h)

            # Convert boxes to integers
            boxes_orig = boxes_orig.astype(int)

            # Get aligned faces for InceptionResnetV1 from original images
            aligned_faces = mtcnn.extract(img_rgb_resized, boxes_orig, save_path=None)

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
                # landmarks_orig[i] has shape (5, 2)
                landmark = landmarks_orig[i]
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
                face_aligned = face_align.norm_crop(img_rgb_resized, landmark=landmark_np, image_size=112)

                # Convert the image to BGR
                face_aligned_bgr = convert_bgr_to_rgb(face_aligned)

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


    # Directory or single image processing logic remains the same
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

        logger.info(f"Processing {len(image_files)} images in directory {args.directory}.")

        if args.batch:
            # Process images in batch
            batch_size = 32  # Adjust based on your GPU memory
            for i in range(0, len(image_files), batch_size):
                batch_paths = image_files[i:i+batch_size]
                process_images_batch(batch_paths)
        else:
            # Process images one by one
            for image_path in image_files:
                process_image(image_path)

    # Calculate processing time and output statistics (unchanged)
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
