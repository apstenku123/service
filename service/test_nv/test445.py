# file: test445.py
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
    parser.add_argument('-t', '--turnoff', choices=['h', 'm', 'd'], nargs='*',
                        help='Turn off specific face detection methods: h=HOG, m=MTCNN, d=DL')
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

    # Initialize face detection method flags
    hog_enabled = True
    dl_enabled = True
    mtcnn_enabled = True

    if args.turnoff:
        if 'h' in args.turnoff:
            hog_enabled = False
        if 'm' in args.turnoff:
            mtcnn_enabled = False
        if 'd' in args.turnoff:
            dl_enabled = False

    if not (hog_enabled or dl_enabled or mtcnn_enabled):
        logger.error("All face detection methods are turned off. At least one must be enabled.")
        sys.exit(1)

    # Initialize MTCNN with keep_all=True
    if mtcnn_enabled:
        mtcnn = MTCNN(keep_all=True, device=device)

    # Initialize InceptionResnetV1 model
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Initialize InsightFace model
    app = FaceAnalysis(allowed_modules=['detection', 'recognition', 'landmark_2d_106'],
                       providers=['CUDAExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider'])
    app.prepare(ctx_id=0 if device.type == 'cuda' else -1, det_size=(640, 640))

    # Get the recognition model
    recognizer = app.models['recognition']

    # Initialize HOG-based face detector
    if hog_enabled:
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        else:
            # Fallback: specify the path manually
            cascade_path = 'haarcascade_frontalface_default.xml'  # Ensure the file is in the same directory
            # Alternatively, specify the full path
            # cascade_path = '/path/to/haarcascade_frontalface_default.xml'

        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            hog_face_detector = cv2.cuda_CascadeClassifier.create(cascade_path)
        else:
            hog_face_detector = cv2.CascadeClassifier(cascade_path)

    # Initialize DL-based face detector
    if dl_enabled:
        modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "deploy.prototxt"

        if not os.path.exists(modelFile) or not os.path.exists(configFile):
            logger.error("DL model files not found. Please ensure 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000_fp16.caffemodel' are present.")
            sys.exit(1)

        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

        # Set the preferable backend and target
        if device.type == 'cuda':
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Store the start time after models are loaded
    processing_start_time = time.time()

    # Initialize statistics counters
    images_processed = 0
    total_images = 0
    total_faces_detected = 0
    total_embeddings_facenet = 0
    total_embeddings_insightface = 0

    # Target size for resizing images for MTCNN
    mtcnn_target_size = (1280, 960)  # Width x Height

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

    # Function to detect faces using HOG with OpenCV and CUDA
    def detect_faces_hog_cuda(image):
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # Upload the image to GPU
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(image)

            # Convert to grayscale on GPU
            gpu_gray = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)

            # Detect faces on GPU
            faces_buffer = cv2.cuda_GpuMat()

            hog_face_detector.detectMultiScale(gpu_gray, faces_buffer)

            # Download faces data to CPU
            faces = faces_buffer.download()

            if faces is not None:
                faces = faces.astype(int)
            else:
                faces = []

        else:
            # Use CPU-based detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = hog_face_detector.detectMultiScale(gray)

        return faces

    # Function to detect faces using DL with OpenCV DNN and CUDA
    def detect_faces_dl_cuda(image):
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        boxes = []
        confidences = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                boxes.append(box.astype(int))
                confidences.append(confidence)

        return boxes, confidences

    # Function to process a single image
    def process_image(image_path):
        nonlocal images_processed, total_images, total_faces_detected, total_embeddings_facenet, total_embeddings_insightface

        # Read the original image using OpenCV
        img_bgr_orig = cv2.imread(image_path)
        if img_bgr_orig is None:
            logger.error(f"Failed to load image: {image_path}")
            return

        # Convert the original image to RGB
        img_rgb_orig = cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2RGB)

        faces = None
        landmarks = None

        # Attempt face detection with HOG
        if hog_enabled:
            faces = detect_faces_hog_cuda(img_bgr_orig)
            if faces is not None and len(faces) > 0:
                logger.debug(f"Faces detected using HOG: {len(faces)}")
                detection_method = 'HOG'
            else:
                faces = None

        # If no faces found and DL is enabled
        if (faces is None or len(faces) == 0) and dl_enabled:
            faces, confidences = detect_faces_dl_cuda(img_bgr_orig)
            if faces is not None and len(faces) > 0:
                logger.debug(f"Faces detected using DL: {len(faces)}")
                detection_method = 'DL'
            else:
                faces = None

        # If no faces found and MTCNN is enabled
        if (faces is None or len(faces) == 0) and mtcnn_enabled:
            # Resize image for MTCNN
            img_rgb_resized, scale_w, scale_h = resize_image_for_mtcnn(img_rgb_orig, mtcnn_target_size)
            # Detect faces with MTCNN on resized image
            boxes_resized, probs_resized, landmarks_resized = mtcnn.detect(img_rgb_resized, landmarks=True)

            if boxes_resized is not None and landmarks_resized is not None:
                num_faces = boxes_resized.shape[0]
                # Adjust boxes and landmarks back to original image coordinates
                boxes_orig, landmarks_orig = adjust_boxes_landmarks(boxes_resized, landmarks_resized, scale_w, scale_h)
                boxes_orig = boxes_orig.astype(int)
                faces = boxes_orig
                landmarks = landmarks_orig
                detection_method = 'MTCNN'
                logger.debug(f"Faces detected using MTCNN: {num_faces}")
            else:
                faces = None

        images_processed += 1
        total_images += 1

        if faces is not None and len(faces) > 0:
            num_faces = len(faces)
            total_faces_detected += num_faces

            # If landmarks are not provided by the detector (HOG or DL), attempt to get them
            if landmarks is None:
                # Use InsightFace to get landmarks
                face_images = []
                for face_rect in faces:
                    x, y, w, h = face_rect
                    face_img = img_rgb_orig[y:y+h, x:x+w]
                    face_images.append(face_img)

                # Get face embeddings and landmarks using InsightFace
                if face_images:
                    embeddings_insightface = []
                    embeddings_facenet = []
                    for face_img in face_images:
                        faces_insight = app.get(face_img)
                        if faces_insight:
                            face = faces_insight[0]
                            # Get embedding with InsightFace
                            embedding_insight = face.embedding
                            embeddings_insightface.append(embedding_insight)
                            total_embeddings_insightface += 1

                            if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                                landmark_106 = face.landmark_2d_106
                                indices = [96, 97, 54, 76, 82]
                                landmarks_face = landmark_106[indices, :]
                            elif hasattr(face, 'kps') and face.kps is not None:
                                landmarks_face = face.kps
                            else:
                                logger.error("No landmarks found for face.")
                                continue

                            # Prepare face for Facenet model
                            face_aligned = face_align.norm_crop(img_rgb_orig, landmark=landmarks_face, image_size=112)
                            # face_aligned = face_align.norm_crop(img_rgb_orig, landmark=face.landmark_2d_106, image_size=160)
                            face_aligned_tensor = torch.from_numpy(face_aligned).permute(2, 0, 1).float().unsqueeze(0)
                            face_aligned_tensor = face_aligned_tensor.to(device)
                            # Get embedding from Facenet
                            with torch.no_grad():
                                embedding_facenet = facenet_model(face_aligned_tensor).cpu().numpy()
                            embeddings_facenet.append(embedding_facenet)
                            total_embeddings_facenet += 1
                        else:
                            logger.error("Failed to get landmarks with InsightFace.")
                            continue

                    logger.debug(f'Number of embeddings from InceptionResnetV1: {len(embeddings_facenet)}')
                    logger.debug(f'Number of embeddings from InsightFace: {len(embeddings_insightface)}')

                else:
                    logger.error("No face images to process for embeddings.")
                    return
            else:
                # Process faces with provided landmarks (from MTCNN)
                # Convert boxes to integers
                boxes_orig = faces.astype(int)
                # Get aligned faces for InceptionResnetV1 from original image
                aligned_faces = mtcnn.extract(img_rgb_orig, boxes_orig, save_path=None)

                if aligned_faces is not None and len(aligned_faces) > 0:
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
            logger.debug(f"No faces detected in image: {os.path.basename(image_path)}")

        # Clear GPU cache every 400 images
        if images_processed % 400 == 0:
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared.")

    # Function to process images in batch (updated accordingly)
    def process_images_batch(image_paths):
        for image_path in image_paths:
            process_image(image_path)

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

    # Calculate processing time
    processing_end_time = time.time()
    total_processing_time = processing_end_time - processing_start_time

    # Calculate images per second
    if total_processing_time > 0:
        images_per_second = total_images / total_processing_time
        faces_per_second = total_faces_detected / total_processing_time
    else:
        images_per_second = 0
        faces_per_second = 0

    # Output statistics
    logger.info("Processing completed.")
    logger.info(f"Total processing time (excluding model loading): {total_processing_time:.2f} seconds")
    logger.info(f"Total images processed: {total_images}")
    logger.info(f"Images per second: {images_per_second:.2f}")
    logger.info(f"Total faces detected: {total_faces_detected}")
    logger.info(f"Faces per second: {faces_per_second:.2f}")
    logger.info(f"Total embeddings from InceptionResnetV1: {total_embeddings_facenet}")
    logger.info(f"Total embeddings from InsightFace: {total_embeddings_insightface}")


if __name__ == '__main__':
    main()
