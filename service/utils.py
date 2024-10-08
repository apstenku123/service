# file: utils.py
# directory: .

import os
import logging
import cv2
import requests
import torch
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from logging.handlers import RotatingFileHandler
from models import Base


def get_engine():
    DB_HOST = os.environ.get('DB_HOST')
    DB_NAME = os.environ.get('DB_NAME')
    DB_USER = os.environ.get('DB_USER')
    DB_PASSWORD = os.environ.get('DB_PASSWORD')

    DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
    engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
    return engine


def get_session_factory(engine):
    return sessionmaker(bind=engine)


def setup_database(engine):
    Base.metadata.create_all(engine)
    # Создаём расширение vector, если его нет

    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()


def preprocess_image(image_path):
    """
    Loads and processes an image.
    """
    image = cv2.imread(image_path)
    if image is not None:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image
    return None


def configure_thread_logging(logger_name, log_filename, log_level, log_output):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    handlers = []
    if log_output in ('file', 'both'):
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        file_handler = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    if log_output in ('console', 'both'):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    # Remove existing handlers
    logger.handlers = []
    for handler in handlers:
        logger.addHandler(handler)

    return logger


def download_image(image_url, local_path, logger, stats_collector):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Downloaded image: {image_url}")
        stats_collector.increment_files_downloaded()
        return True
    except Exception as e:
        logger.error(f"Error downloading {image_url}: {e}")
        return False

def get_face_analysis_model(device):
    """
    Returns a FaceAnalysis model from insightface.app.
    """
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(providers=['CUDAExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider'])
    app.prepare(ctx_id=0 if device.type == 'cuda' else -1)
    return app

def get_insightface_embedding(image, app):
    """
    Генерирует эмбеддинг с помощью FaceAnalysis из insightface.app.
    """
    faces = app.get(image)
    if faces:
        # Берем первое обнаруженное лицо
        face = faces[0]
        embedding = face.embedding.flatten()
        return embedding
    else:
        return None

def get_facenet_embedding(image, mtcnn, model, device):
    """
    Генерирует эмбеддинг с помощью MTCNN и InceptionResnetV1 из facenet_pytorch.
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face = mtcnn(img_rgb)
    if face is not None:
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(face).cpu().numpy().flatten()
        return embedding
    else:
        return None