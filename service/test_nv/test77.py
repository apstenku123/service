# file: test77.py
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

    # Number of processing slots (streams)
    num_slots = 2  # Adjust based on your GPU's capability

    # Define ProcessingSlot class
    class ProcessingSlot:
        def __init__(self):
            self.stream = cv2.cuda_Stream()
            # Pinned host memory for input images
            self.img_bgr_orig = None
            self.img_rgb_orig = None
            # Device buffers
            self.gpu_img = cv2.cuda_GpuMat()
            self.gpu_rgb = cv2.cuda_GpuMat()
            self.gpu_resized = cv2.cuda_GpuMat()
            # Pre-allocated arrays for images
            self.img_rgb_resized = None
            # Variables to store intermediate results
            self.scale_w = None
            self.scale_h = None
            self.boxes_resized = None
            self.probs_resized = None
            self.landmarks_resized = None
            self.boxes_orig = None
            self.landmarks_orig = None
            self.image_path = None

        def release(self):
            # Unregister pinned memory
            if self.img_bgr_orig is not None:
                cv2.cuda.unregisterPageLocked(self.img_bgr_orig)
                self.img_bgr_orig = None
            if self.img_rgb_orig is not None:
                cv2.cuda.unregisterPageLocked(self.img_rgb_orig)
                self.img_rgb_orig = None
            if self.img_rgb_resized is not None:
                cv2.cuda.unregisterPageLocked(self.img_rgb_resized)
                self.img_rgb_resized = None

    # Initialize processing slots
    processing_slots = [ProcessingSlot() for _ in range(num_slots)]

    # Function to resize image for MTCNN and compute scaling factors using CUDA
    def resize_image_for_mtcnn(slot, target_size):
        img_rgb_orig = slot.img_rgb_orig
        original_size = img_rgb_orig.shape[:2]  # (height, width)
        h, w = original_size
        target_w, target_h = target_size

        # Calculate scaling factors
        scale_w = target_w / w
        scale_h = target_h / h

        # Resize image using CUDA
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # Upload img_rgb_orig to GPU if not already uploaded
            if slot.gpu_rgb.empty():
                slot.gpu_rgb.upload(img_rgb_orig, stream=slot.stream)
            cv2.cuda.resize(slot.gpu_rgb, (target_w, target_h), dst=slot.gpu_resized,
                            interpolation=cv2.INTER_LINEAR, stream=slot.stream)
            # Download into pre-allocated array
            slot.gpu_resized.download(dst=slot.img_rgb_resized, stream=slot.stream)
            slot.stream.waitForCompletion()
        else:
            slot.img_rgb_resized = cv2.resize(img_rgb_orig, (target_w, target_h))

        return scale_w, scale_h

    # Function to convert BGR to RGB using CUDA
    def convert_bgr_to_rgb(slot):
        img_bgr_orig = slot.img_bgr_orig
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            slot.gpu_img.upload(img_bgr_orig, stream=slot.stream)
            cv2.cuda.cvtColor(slot.gpu_img, cv2.COLOR_BGR2RGB, dst=slot.gpu_rgb, stream=slot.stream)
            # Download into pre-allocated array
            slot.gpu_rgb.download(dst=slot.img_rgb_orig, stream=slot.stream)
            slot.stream.waitForCompletion()
        else:
            slot.img_rgb_orig[:] = cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2RGB)

    # Function to adjust boxes and landmarks back to original image coordinates
    def adjust_boxes_landmarks(boxes, landmarks, scale_w, scale_h):
        # Adjust boxes
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / scale_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / scale_h

        # Adjust landmarks
        landmarks[:, :, 0] = landmarks[:, :, 0] / scale_w
        landmarks[:, :, 1] = landmarks[:, :, 1] / scale_h

        return boxes, landmarks

    # Function to process an image in a processing slot
    def process_image_in_slot(slot):
        nonlocal images_processed, total_images, total_faces_mtcnn, total_embeddings_facenet, total_embeddings_insightface

        # Convert the original image to RGB
        convert_bgr_to_rgb(slot)

        # Resize image for MTCNN
        scale_w, scale_h = resize_image_for_mtcnn(slot, mtcnn_target_size)

        # Detect faces with MTCNN on resized image
        boxes_resized, probs_resized, landmarks_resized = mtcnn.detect(slot.img_rgb_resized, landmarks=True)

        images_processed += 1
        total_images += 1

        if boxes_resized is not None and landmarks_resized is not None:
            num_faces = boxes_resized.shape[0]
            total_faces_mtcnn += num_faces
            logger.debug(f'Image: {os.path.basename(slot.image_path)} - Number of faces detected by MTCNN: {num_faces}')

            # Adjust boxes and landmarks back to original image coordinates
            boxes_orig, landmarks_orig = adjust_boxes_landmarks(boxes_resized, landmarks_resized, scale_w, scale_h)

            # Convert boxes to integers
            boxes_orig = boxes_orig.astype(int)

            # Get aligned faces for InceptionResnetV1 from original image
            aligned_faces = mtcnn.extract(slot.img_rgb_orig, boxes_orig, save_path=None)

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
                face_aligned = face_align.norm_crop(slot.img_rgb_orig, landmark=landmark_np, image_size=112)

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
            logger.debug(f'MTCNN did not detect faces in image: {os.path.basename(slot.image_path)}')

        # Clear GPU cache every 400 images
        if images_processed % 400 == 0:
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared.")

    if args.directory:
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

        image_index = 0
        total_images_count = len(image_files)

        while image_index < total_images_count:
            for slot in processing_slots:
                if image_index >= total_images_count:
                    break
                image_path = image_files[image_index]

                # If not the first loop, wait for the previous operations in this slot to complete
                if image_index >= num_slots:
                    slot.stream.waitForCompletion()
                    slot.release()

                # Read image into pinned host memory
                img_bgr_orig = cv2.imread(image_path)
                if img_bgr_orig is None:
                    logger.error(f"Failed to load image: {image_path}")
                    image_index += 1
                    continue

                # Register pinned memory for input image
                cv2.cuda.registerPageLocked(img_bgr_orig)

                # Initialize empty arrays with pinned memory
                img_rgb_orig = np.empty_like(img_bgr_orig)
                cv2.cuda.registerPageLocked(img_rgb_orig)

                # Pre-allocate the resized image array based on target size
                target_w, target_h = mtcnn_target_size
                img_rgb_resized = np.empty((target_h, target_w, 3), dtype=img_rgb_orig.dtype)
                cv2.cuda.registerPageLocked(img_rgb_resized)

                # Set slot variables
                slot.img_bgr_orig = img_bgr_orig
                slot.img_rgb_orig = img_rgb_orig
                slot.img_rgb_resized = img_rgb_resized
                slot.image_path = image_path

                # Start processing the image in the slot
                process_image_in_slot(slot)

                image_index += 1

        # After processing all images, wait for remaining operations to complete
        for slot in processing_slots:
            slot.stream.waitForCompletion()
            slot.release()

    elif args.image_file:
        if not os.path.exists(args.image_file):
            logger.error(f"Image file {args.image_file} does not exist.")
            sys.exit(1)

        # Since we're processing a single image, we don't need multiple streams
        slot = ProcessingSlot()
        slot.stream = cv2.cuda_Stream()
        image_path = args.image_file

        # Read image into pinned host memory
        img_bgr_orig = cv2.imread(image_path)
        if img_bgr_orig is None:
            logger.error(f"Failed to load image: {image_path}")
            sys.exit(1)

        # Register pinned memory
        cv2.cuda.registerPageLocked(img_bgr_orig)

        # Initialize empty array for img_rgb_orig with pinned memory
        img_rgb_orig = np.empty_like(img_bgr_orig)
        cv2.cuda.registerPageLocked(img_rgb_orig)

        # Pre-allocate the resized image array based on target size
        target_w, target_h = mtcnn_target_size
        img_rgb_resized = np.empty((target_h, target_w, 3), dtype=img_rgb_orig.dtype)
        cv2.cuda.registerPageLocked(img_rgb_resized)

        # Set slot variables
        slot.img_bgr_orig = img_bgr_orig
        slot.img_rgb_orig = img_rgb_orig
        slot.img_rgb_resized = img_rgb_resized
        slot.image_path = image_path

        # Process the image
        process_image_in_slot(slot)

        # Wait for the operation to complete and release resources
        slot.stream.waitForCompletion()
        slot.release()

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
