# file: utils.py
# directory: .
import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import cv2
import torch
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
