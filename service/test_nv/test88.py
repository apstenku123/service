# file: test88.py
# directory: test_nv
import argparse
import os
import sys
import time
import logging
import torch
import cv2
import cupy as cp
import numpy as np

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
    mtcnn_target_size = (1280, 960)  # Width x Height

    # Function to resize image for MTCNN and compute scaling factors
    def resize_image_for_mtcnn(image_gpu, target_size, original_size):
        h, w = original_size
        target_w, target_h = target_size

        # Calculate scaling factors
        scale_w = target_w / w
        scale_h = target_h / h

        # Resize image on GPU
        resized_image_gpu = cv2.cuda.resize(image_gpu, (target_w, target_h))

        return resized_image_gpu, scale_w, scale_h

    # Function to convert cv2.cuda_GpuMat to torch.Tensor
    def gpu_mat_to_tensor(gpu_mat):
        # Download data from GpuMat to numpy array
        cpu_array = gpu_mat.download()
        # Convert numpy array to torch.Tensor and move to GPU
        tensor = torch.from_numpy(cpu_array).permute(2, 0, 1).float().to(device)
        return tensor / 255.0  # Normalize

    # Function to get aligned faces on GPU (simplified version)
    def align_faces_on_gpu(img_tensor, boxes):
        faces = []
        for box in boxes:
            # Compute coordinates for face cropping
            x1, y1, x2, y2 = box.int()
            face_tensor = img_tensor[:, y1:y2, x1:x2]

            # Resize face tensor to (160, 160) as expected by facenet_model
            face_tensor = torch.nn.functional.interpolate(face_tensor.unsqueeze(0), size=(160, 160), mode='bilinear', align_corners=False)
            faces.append(face_tensor.squeeze(0))

        if faces:
            faces_tensor = torch.stack(faces).to(device)
            return faces_tensor
        else:
            return None

    # Function to process a single image
    def process_image(image_path):
        nonlocal images_processed, total_images, total_faces_mtcnn, total_embeddings_facenet, total_embeddings_insightface

        # Read the image on CPU to get the size
        img_cpu = cv2.imread(image_path)
        if img_cpu is None:
            logger.error(f"Failed to load image: {image_path}")
            return

        # Get original image size
        h, w = img_cpu.shape[:2]

        # Upload image to GPU
        img_gpu = cv2.cuda_GpuMat()
        img_gpu.upload(img_cpu)

        # Convert color on GPU
        img_rgb_gpu = cv2.cuda.cvtColor(img_gpu, cv2.COLOR_BGR2RGB)

        # Resize image for MTCNN on GPU
        img_rgb_resized_gpu, scale_w, scale_h = resize_image_for_mtcnn(img_rgb_gpu, mtcnn_target_size, (h, w))

        # Convert to torch.Tensor on GPU
        img_rgb_resized_tensor = gpu_mat_to_tensor(img_rgb_resized_gpu)

        # Detect faces with MTCNN on GPU
        boxes_resized, probs_resized, landmarks_resized = mtcnn.detect(img_rgb_resized_tensor.permute(1, 2, 0), landmarks=True)

        images_processed += 1
        total_images += 1

        if boxes_resized is not None and landmarks_resized is not None:
            num_faces = boxes_resized.shape[0]
            total_faces_mtcnn += num_faces
            logger.debug(f'Image: {os.path.basename(image_path)} - Number of faces detected by MTCNN: {num_faces}')

            # Adjust boxes and landmarks back to original image coordinates
            boxes_resized = torch.from_numpy(boxes_resized).to(device)
            landmarks_resized = torch.from_numpy(landmarks_resized).to(device)

            boxes_orig = boxes_resized.clone()
            boxes_orig[:, [0, 2]] = boxes_orig[:, [0, 2]] / scale_w
            boxes_orig[:, [1, 3]] = boxes_orig[:, [1, 3]] / scale_h

            landmarks_orig = landmarks_resized.clone()
            landmarks_orig[:, :, 0] = landmarks_orig[:, :, 0] / scale_w
            landmarks_orig[:, :, 1] = landmarks_orig[:, :, 1] / scale_h

            # Convert original image to tensor
            img_rgb_tensor = gpu_mat_to_tensor(img_rgb_gpu)

            # Get aligned faces on GPU
            aligned_faces = align_faces_on_gpu(img_rgb_tensor, boxes_orig)

            if aligned_faces is not None and len(aligned_faces) > 0:
                # Get embeddings from InceptionResnetV1
                with torch.no_grad():
                    embeddings_facenet = facenet_model(aligned_faces).cpu().numpy()
                num_embeddings_facenet = embeddings_facenet.shape[0]
                total_embeddings_facenet += num_embeddings_facenet
                logger.debug(f'Number of embeddings from InceptionResnetV1: {num_embeddings_facenet}')
            else:
                logger.error("Failed to get aligned faces for InceptionResnetV1.")
                return

            # Get embeddings from InsightFace
            embeddings_insightface = []
            img_rgb_orig_cpu = img_rgb_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            for i in range(num_faces):
                # landmarks_orig[i] has shape (5, 2)
                landmark = landmarks_orig[i].cpu().numpy()
                # Use face_align.norm_crop on CPU
                face_aligned = face_align.norm_crop(img_rgb_orig_cpu, landmark=landmark, image_size=112)

                # Convert the image to BGR
                face_aligned_bgr = cv2.cvtColor(face_aligned, cv2.COLOR_RGB2BGR)

                # Convert to CuPy array
                face_aligned_bgr_cp = cp.asarray(face_aligned_bgr.astype('uint8'))

                # Get embedding with recognizer
                embedding_insight = recognizer.get_feat(face_aligned_bgr_cp)
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

        for image_path in image_files:
            process_image(image_path)

    # Calculate processing time and output statistics
    processing_end_time = time.time()
    total_processing_time = processing_end_time - processing_start_time

    if total_processing_time > 0:
        images_per_second = total_images / total_processing_time
        faces_per_second = total_faces_mtcnn / total_processing_time
    else:
        images_per_second = 0
        faces_per_second = 0

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
