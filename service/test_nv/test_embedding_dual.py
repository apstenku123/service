# file: test_embedding_dual.py
# directory: test_nv

import argparse
import os
import logging
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from utils import (
    get_engine,
    get_session_factory,
    configure_thread_logging,
    get_face_analysis_model,
    get_insightface_embedding,
    get_facenet_embedding
)
from models import Base, ImageEmbedding, BaseImageUrl, Image
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker


def main():
    parser = argparse.ArgumentParser(
        description='Тестовый скрипт для создания эмбеддингов из изображения двумя методами.')
    parser.add_argument('image_path', type=str, help='Путь к файлу изображения.')
    parser.add_argument('--log-level', type=str, default=os.environ.get('LOG_LEVEL', 'INFO'),
                        help='Уровень логирования (по умолчанию INFO)')
    parser.add_argument('--log-output', type=str, choices=['file', 'console', 'both'],
                        default=os.environ.get('LOG_OUTPUT', 'console'),
                        help='Вывод логов: file, console или both (по умолчанию console)')

    args = parser.parse_args()

    image_path = args.image_path
    LOG_LEVEL = getattr(logging, args.log_level.upper(), logging.INFO)
    LOG_OUTPUT = args.log_output

    # Настройка логирования
    log_filename = 'logs/test_embedding_dual.log'
    logger = configure_thread_logging('test_embedding_dual', log_filename, LOG_LEVEL, LOG_OUTPUT)

    # Проверка существования файла изображения
    if not os.path.isfile(image_path):
        logger.error(f"Файл изображения {image_path} не существует.")
        return

    # Инициализация устройства
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используется устройство: {device}")

    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Не удалось загрузить изображение {image_path}.")
        return

    # Инициализация моделей
    app = get_face_analysis_model(device)
    mtcnn = MTCNN(keep_all=False, device=device)
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Генерация эмбеддингов
    insightface_embedding = get_insightface_embedding(img, app)
    facenet_embedding = get_facenet_embedding(img, mtcnn, facenet_model, device)

    if insightface_embedding is None and facenet_embedding is None:
        logger.error("Не удалось сгенерировать эмбеддинги из изображения.")
        return

    # Подключение к базе данных
    engine = get_engine()
    SessionFactory = get_session_factory(engine)
    session = SessionFactory()

    # Убеждаемся, что таблицы существуют
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    try:
        # Создаем или получаем запись BaseImageUrl и Image
        base_url_str = "http://test.base.url"
        base_url = session.query(BaseImageUrl).filter_by(base_url=base_url_str).first()
        if not base_url:
            base_url = BaseImageUrl(base_url=base_url_str)
            session.add(base_url)
            session.commit()

        image_filename = os.path.basename(image_path)
        image = Image(base_url_id=base_url.id, filename=image_filename)
        session.add(image)
        session.commit()

        # Создаем запись ImageEmbedding
        embedding_record = ImageEmbedding(
            image_id=image.id,
            filename=image_filename,
            embedding=facenet_embedding.tolist() if facenet_embedding is not None else None,
            insightface_embedding=insightface_embedding.tolist() if insightface_embedding is not None else None
        )
        session.add(embedding_record)
        session.commit()
        logger.info("Эмбеддинги сохранены в базе данных.")

    except Exception as e:
        session.rollback()
        logger.error(f"Ошибка при сохранении эмбеддингов в базе данных: {e}", exc_info=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
