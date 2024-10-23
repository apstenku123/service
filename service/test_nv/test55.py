# file: test55.py
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

    # Function to resize image for MTCNN and compute scaling factors using CUDA
    def resize_image_for_mtcnn(image, target_size):
        original_size = image.shape[:2]  # (height, width)
        h, w = original_size
        target_w, target_h = target_size

        # Calculate scaling factors
        scale_w = target_w / w
        scale_h = target_h / h

        # Resize image using CUDA if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)
            gpu_resized = cv2.cuda.resize(gpu_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            resized_image = gpu_resized.download()
        else:
            resized_image = cv2.resize(image, (target_w, target_h))

        return resized_image, scale_w, scale_h

    # Function to convert BGR to RGB using CUDA
    def convert_bgr_to_rgb(image):
        # Convert color space using CUDA if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)
            gpu_rgb = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2RGB)
            img_rgb = gpu_rgb.download()
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return img_rgb

    # Function to adjust boxes and landmarks back to original image coordinates
    def adjust_boxes_landmarks(boxes, landmarks, scale_w, scale_h):
        # Adjust boxes
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / scale_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / scale_h

        # Adjust landmarks
        landmarks[:, :, 0] = landmarks[:, :, 0] / scale_w
        landmarks[:, :, 1] = landmarks[:, :, 1] / scale_h

        return boxes, landmarks

    # Function to process a single image
    def process_image(image_path):
        nonlocal images_processed, total_images, total_faces_mtcnn, total_embeddings_facenet, total_embeddings_insightface

        # Read the original image using OpenCV
        img_bgr_orig = cv2.imread(image_path)
        if img_bgr_orig is None:
            logger.error(f"Failed to load image: {image_path}")
            return

        # Convert the original image to RGB
        img_rgb_orig = convert_bgr_to_rgb(img_bgr_orig)

        # Resize image for MTCNN
        img_rgb_resized, scale_w, scale_h = resize_image_for_mtcnn(img_rgb_orig, mtcnn_target_size)

        # Detect faces with MTCNN on resized image
        boxes_resized, probs_resized, landmarks_resized = mtcnn.detect(img_rgb_resized, landmarks=True)

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

            # Get aligned faces for InceptionResnetV1 from original image
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
                face_aligned = face_align.norm_crop(img_rgb_orig, landmark=landmark_np, image_size=112)

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

    # Function to process images in batch
    def process_images_batch(image_paths):
        nonlocal images_processed, total_images, total_faces_mtcnn, total_embeddings_facenet, total_embeddings_insightface

        # Read and preprocess all images
        images_orig = []
        images_resized = []
        scales_w = []
        scales_h = []
        valid_image_paths = []
        for image_path in image_paths:
            img_bgr_orig = cv2.imread(image_path)
            if img_bgr_orig is None:
                logger.error(f"Failed to load image: {image_path}")
                continue

            # Convert original image to RGB
            img_rgb_orig = convert_bgr_to_rgb(img_bgr_orig)

            # Resize image for MTCNN
            img_rgb_resized, scale_w, scale_h = resize_image_for_mtcnn(img_rgb_orig, mtcnn_target_size)

            images_orig.append(img_rgb_orig)
            images_resized.append(img_rgb_resized)
            scales_w.append(scale_w)
            scales_h.append(scale_h)
            valid_image_paths.append(image_path)

        if not images_resized:
            logger.error("No valid images to process.")
            return

        # Detect faces with MTCNN in batch on resized images
        boxes_batch_resized, probs_batch_resized, landmarks_batch_resized = mtcnn.detect(images_resized, landmarks=True)

        for idx, image_path in enumerate(valid_image_paths):
            img_rgb_orig = images_orig[idx]
            img_rgb_resized = images_resized[idx]
            scale_w = scales_w[idx]
            scale_h = scales_h[idx]

            boxes_resized = boxes_batch_resized[idx]
            landmarks_resized = landmarks_batch_resized[idx]

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
            # Split image_files into chunks to avoid memory issues
            batch_size = 32  # Adjust based on your GPU memory
            for i in range(0, len(image_files), batch_size):
                batch_paths = image_files[i:i+batch_size]
                process_images_batch(batch_paths)
        else:
            # Process images one by one
            for image_path in image_files:
                process_image(image_path)

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
