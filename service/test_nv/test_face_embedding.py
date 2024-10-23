# file: test_face_embedding.py
# directory: test_nv

import argparse
import os
import logging
import cv2
import torch

from utils import get_engine, get_session_factory, configure_thread_logging, get_face_analysis_model
from models import Base, TestEmbedding
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker


def main():
    parser = argparse.ArgumentParser(description='Test script for creating embeddings from an image.')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    parser.add_argument('--log-level', type=str, default=os.environ.get('LOG_LEVEL', 'INFO'),
                        help='Logging level (default INFO)')
    parser.add_argument('--log-output', type=str, choices=['file', 'console', 'both'],
                        default=os.environ.get('LOG_OUTPUT', 'console'),
                        help='Logging output: file, console, or both (default console)')

    args = parser.parse_args()

    image_path = args.image_path
    LOG_LEVEL = getattr(logging, args.log_level.upper(), logging.INFO)
    LOG_OUTPUT = args.log_output

    # Configure logging
    log_filename = 'logs/test_embedding.log'
    logger = configure_thread_logging('test_embedding', log_filename, LOG_LEVEL, LOG_OUTPUT)

    # Check if image file exists
    if not os.path.isfile(image_path):
        logger.error(f"Image file {image_path} does not exist.")
        return

    # Initialize device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Get FaceAnalysis model
    app = get_face_analysis_model(device)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image {image_path}.")
        return

    # Detect faces and get embeddings
    faces = app.get(img)
    if not faces:
        logger.info(f"No faces detected in image {image_path}.")
        return

    embeddings = [face.embedding.flatten().tolist() for face in faces]

    # Connect to database
    engine = get_engine()
    SessionFactory = get_session_factory(engine)
    session = SessionFactory()

    # Ensure table exists
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Save embeddings to database
    try:
        for idx, embedding_vector in enumerate(embeddings):
            embedding = TestEmbedding(
                filename=os.path.basename(image_path),
                embedding=embedding_vector
            )
            session.add(embedding)
        session.commit()
        logger.info(f"Saved {len(embeddings)} embeddings to the database.")
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving embeddings to database: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
