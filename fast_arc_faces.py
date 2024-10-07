import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import argparse
import logging
import threading
from queue import Queue
from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, DateTime, UniqueConstraint
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import IntegrityError
from datetime import datetime
import torch
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import traceback
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Конфигурация логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Базовый класс для SQLAlchemy ORM
Base = declarative_base()

# Определение ORM моделей
class BaseImageUrl(Base):
    __tablename__ = 'base_image_urls'
    id = Column(Integer, primary_key=True)
    base_url = Column(String, unique=True, nullable=False)
    images = relationship("Image", back_populates="base_image_url")

class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True)
    base_url_id = Column(Integer, ForeignKey('base_image_urls.id'), nullable=False)
    image_path = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    processed = Column(Boolean, default=False)
    base_image_url = relationship("BaseImageUrl", back_populates="images")
    embeddings = relationship("ImageEmbedding", back_populates="image")

    __table_args__ = (UniqueConstraint('base_url_id', 'image_path', name='_base_image_uc'),)

class Batch(Base):
    __tablename__ = 'batches'
    id = Column(Integer, primary_key=True)
    page_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.now())
    processed = Column(Boolean, default=False)
    images = relationship("BatchImage", back_populates="batch")

    __table_args__ = (UniqueConstraint('page_number', name='_page_number_uc'),)

class BatchImage(Base):
    __tablename__ = 'batch_images'
    batch_id = Column(Integer, ForeignKey('batches.id'), primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), primary_key=True)
    batch = relationship("Batch", back_populates="images")
    image = relationship("Image")

class Checkpoint(Base):
    __tablename__ = 'checkpoints'
    id = Column(Integer, primary_key=True)
    page_url = Column(String, unique=True, nullable=False)

class ImageEmbedding(Base):
    __tablename__ = 'image_embeddings'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'))
    filename = Column(String, nullable=False)
    embedding = Column(String, nullable=False)  # Хранение как строки для простоты
    image = relationship("Image", back_populates="embeddings")

# Функция для подключения к базе данных
def get_engine():
    DB_HOST = os.environ.get('DB_HOST')
    DB_NAME = os.environ.get('DB_NAME')
    DB_USER = os.environ.get('DB_USER')
    DB_PASSWORD = os.environ.get('DB_PASSWORD')

    DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
    engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
    return engine

# Создание сессии фабрики
def get_session_factory(engine):
    return sessionmaker(bind=engine)

# Функция для настройки базы данных
def setup_database(engine):
    Base.metadata.create_all(engine)

# Функция для предобработки изображения
def preprocess_image(image_path):
    """
    Загружает и обрабатывает изображение.
    """
    image = cv2.imread(image_path)
    if image is not None:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image
    return None

# Функция для обработки страницы и извлечения URL изображений
def process_page(session, page_url):
    logger.info(f"Обработка страницы: {page_url}")
    try:
        response = requests.get(page_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Ошибка загрузки страницы {page_url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')

    # Проверка, была ли уже обработана страница
    checkpoint = session.query(Checkpoint).filter_by(page_url=page_url).first()
    if checkpoint:
        logger.info(f"Страница {page_url} уже обработана. Пропуск.")
        return []

    # Добавление checkpoint
    new_checkpoint = Checkpoint(page_url=page_url)
    session.add(new_checkpoint)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        logger.warning(f"Страница {page_url} уже добавлена в checkpoint. Пропуск.")
        return []

    # Извлечение URL изображений
    posts = soup.find_all('div', class_='post-container')
    image_urls = [post.find('img')['src'].replace('.th', '') for post in posts if post.find('img')]

    logger.info(f"Найдено {len(image_urls)} изображений на странице {page_url}")
    return image_urls

# Функция для сохранения изображений и создания батчей
def store_images_and_batches(session, image_urls, page_number):
    if not image_urls:
        return None

    # Определение общего префикса URL
    common_prefix = get_longest_common_prefix(image_urls)
    parsed_url = urlparse(image_urls[0])
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{common_prefix}/"

    # Добавление base_image_url если не существует
    base_image_url = session.query(BaseImageUrl).filter_by(base_url=base_url).first()
    if not base_image_url:
        base_image_url = BaseImageUrl(base_url=base_url)
        session.add(base_image_url)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            base_image_url = session.query(BaseImageUrl).filter_by(base_url=base_url).first()

    image_ids = []
    for img_url in image_urls:
        relative_path = img_url.replace(base_image_url.base_url, '').lstrip('/')
        filename = os.path.basename(relative_path)
        # Проверка, существует ли уже изображение
        image = session.query(Image).filter_by(base_url_id=base_image_url.id, image_path=relative_path).first()
        if not image:
            image = Image(base_url_id=base_image_url.id, image_path=relative_path, filename=filename)
            session.add(image)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                image = session.query(Image).filter_by(base_url_id=base_image_url.id, image_path=relative_path).first()
        image_ids.append(image.id)

    # Создание батча
    batch = Batch(page_number=page_number)
    session.add(batch)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        batch = session.query(Batch).filter_by(page_number=page_number).first()
        if batch.processed:
            logger.info(f"Батч для страницы {page_number} уже обработан.")
            return None

    # Добавление изображений в батч
    for img_id in image_ids:
        batch_image = BatchImage(batch_id=batch.id, image_id=img_id)
        session.add(batch_image)
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        logger.error(f"Ошибка добавления изображений в батч {batch.id}: {e}")

    return batch

# Функция для получения самого длинного общего префикса URL
def get_longest_common_prefix(urls):
    if not urls:
        return ''
    split_urls = [urlparse(url).path.split('/') for url in urls]
    min_length = min(len(splits) for splits in split_urls)
    common_prefix = []
    for i in range(min_length):
        segments = set(splits[i] for splits in split_urls)
        if len(segments) == 1:
            common_prefix.append(segments.pop())
        else:
            break
    return '/'.join(common_prefix)

# Функция для скачивания изображения
def download_image(image_url, local_path):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Скачано изображение: {image_url}")
        return True
    except Exception as e:
        logger.error(f"Ошибка скачивания {image_url}: {e}")
        return False

# Поток скачивания изображений
def downloader_thread(page_queue, batch_queue, download_dir, max_batches_on_disk, engine, download_threads):
    SessionFactory = get_session_factory(engine)
    session = SessionFactory()
    batches_on_disk = 0
    while True:
        if batches_on_disk >= max_batches_on_disk:
            logger.info("Достигнуто максимальное количество батчей на диске. Ожидание освобождения места...")
            time.sleep(5)
            continue

        page_info = page_queue.get()
        if page_info is None:
            page_queue.task_done()
            break  # Получен сигнал завершения

        page_url, page_number = page_info
        image_urls = process_page(session, page_url)
        if image_urls:
            # Создаем батч и сохраняем в БД
            batch = store_images_and_batches(session, image_urls, page_number)
            if batch is None:
                logger.info(f"Батч для страницы {page_number} уже обработан или не содержит изображений.")
                page_queue.task_done()
                continue

            # Создаем временную директорию для батча
            batch_dir = os.path.join(download_dir, f"batch_{batch.id}")
            os.makedirs(batch_dir, exist_ok=True)

            # Скачиваем изображения в батчевую директорию параллельно
            images = session.query(Image).join(BatchImage).filter(BatchImage.batch_id == batch.id).all()
            image_download_futures = []
            with ThreadPoolExecutor(max_workers=download_threads) as executor:
                for img in images:
                    img_url = f"{img.base_image_url.base_url}{img.image_path}"
                    local_path = os.path.join(batch_dir, img.filename)
                    future = executor.submit(download_image, img_url, local_path)
                    image_download_futures.append(future)

                # Ожидаем завершения всех загрузок
                for future in as_completed(image_download_futures):
                    result = future.result()
                    if not result:
                        logger.error(f"Ошибка при скачивании изображения в батче {batch.id}")

            # Помечаем, что батч скачан и готов для обработки
            batch_queue.put((batch.id, batch_dir))
            batches_on_disk += 1
            logger.info(f"Батч {batch.id} добавлен в очередь на обработку.")
        else:
            logger.info(f"Нет изображений на странице {page_url}")
        page_queue.task_done()
    session.close()

# Поток обработки батчей с использованием обработки изображений батчами
def processing_thread(batch_queue, embeddings_queue, model, mtcnn, device, engine, batch_size, report_dir):
    SessionFactory = get_session_factory(engine)
    session = SessionFactory()
    while True:
        batch_info = batch_queue.get()
        if batch_info is None:
            batch_queue.task_done()
            break  # Получен сигнал завершения

        batch_id, batch_dir = batch_info
        logger.info(f"Начало обработки батча {batch_id}")
        embeddings_data = []
        try:
            images = session.query(Image).join(BatchImage).filter(BatchImage.batch_id == batch_id).all()
            image_paths = []
            image_ids = []
            filenames = []
            for img in images:
                local_path = os.path.join(batch_dir, img.filename)
                if not os.path.exists(local_path):
                    logger.error(f"Файл не найден: {local_path}")
                    continue
                image_paths.append(local_path)
                image_ids.append(img.id)
                filenames.append(img.filename)

            # Предобработка изображений (загрузка и конвертация)
            images_data = []
            valid_image_ids = []
            valid_filenames = []
            for idx, path in enumerate(image_paths):
                img_rgb = preprocess_image(path)
                if img_rgb is not None:
                    images_data.append(img_rgb)
                    valid_image_ids.append(image_ids[idx])
                    valid_filenames.append(filenames[idx])
                else:
                    logger.error(f"Не удалось загрузить изображение: {path}")

            total_images = len(images_data)
            total_faces = 0
            images_with_faces = 0
            images_without_faces = 0

            for i in range(0, total_images, batch_size):
                batch_imgs = images_data[i:i + batch_size]
                batch_ids = valid_image_ids[i:i + batch_size]
                batch_filenames = valid_filenames[i:i + batch_size]

                # Детектирование лиц в батче
                try:
                    faces_batch = mtcnn(batch_imgs)
                except Exception as e:
                    logger.error(f"Ошибка детекции лиц в батче: {e}")
                    faces_batch = [None] * len(batch_imgs)

                batch_faces = []
                face_image_ids = []
                face_filenames = []
                for idx, faces in enumerate(faces_batch):
                    if faces is not None:
                        num_faces = faces.shape[0] if len(faces.shape) == 4 else 1
                        total_faces += num_faces
                        images_with_faces += 1
                        # Если несколько лиц в изображении
                        if len(faces.shape) == 4:
                            for j in range(num_faces):
                                batch_faces.append(faces[j])
                                face_image_ids.append(batch_ids[idx])
                                face_filenames.append(batch_filenames[idx])
                        else:
                            batch_faces.append(faces)
                            face_image_ids.append(batch_ids[idx])
                            face_filenames.append(batch_filenames[idx])
                    else:
                        images_without_faces += 1
                        logger.info(f"Лица не обнаружены в изображении ID {batch_ids[idx]}")

                if not batch_faces:
                    continue

                # Обработка эмбеддингов батчем
                faces_tensor = torch.stack(batch_faces).to(device)
                with torch.no_grad():
                    embeddings = model(faces_tensor).cpu().tolist()

                # Сбор эмбеддингов
                for img_id, filename, embedding in zip(face_image_ids, face_filenames, embeddings):
                    embeddings_data.append({'image_id': img_id, 'filename': filename, 'embedding': embedding})

            # Передаем данные для записи в базу данных
            embeddings_queue.put((batch_id, embeddings_data))

            # Генерируем отчёт по батчу
            report = {
                'batch_id': batch_id,
                'total_images': total_images,
                'total_faces': total_faces,
                'images_with_faces': images_with_faces,
                'images_without_faces': images_without_faces
            }
            report_filename = os.path.join(report_dir, f"batch_{batch_id}_report.txt")
            with open(report_filename, 'w') as f:
                f.write(f"Batch ID: {batch_id}\n")
                f.write(f"Total images processed: {total_images}\n")
                f.write(f"Total faces detected: {total_faces}\n")
                f.write(f"Images with faces: {images_with_faces}\n")
                f.write(f"Images without faces: {images_without_faces}\n")
            logger.info(f"Отчёт по батчу {batch_id} сохранён в {report_filename}")

            logger.info(f"Батч {batch_id} обработан и передан для сохранения эмбеддингов.")

        except Exception as e:
            logger.error(f"Ошибка обработки батча {batch_id}: {e}")
            logger.debug(traceback.format_exc())
        finally:
            # Удаляем батчевую директорию
            shutil.rmtree(batch_dir)
            batch_queue.task_done()
        time.sleep(1)  # Добавляем небольшую задержку для снижения нагрузки
    session.close()

# Поток сохранения эмбеддингов
def embeddings_writer_thread(embeddings_queue, engine):
    SessionFactory = get_session_factory(engine)
    session = SessionFactory()
    while True:
        embeddings_info = embeddings_queue.get()
        if embeddings_info is None:
            embeddings_queue.task_done()
            break  # Получен сигнал завершения

        batch_id, embeddings_data = embeddings_info
        logger.info(f"Начало сохранения эмбеддингов для батча {batch_id}")
        try:
            if not embeddings_data:
                logger.info(f"Нет эмбеддингов для сохранения в батче {batch_id}")
                embeddings_queue.task_done()
                continue

            # Сохранение эмбеддингов в базе данных одной транзакцией
            for data in embeddings_data:
                embedding_str = ','.join(map(str, data['embedding']))
                embedding = ImageEmbedding(image_id=data['image_id'], filename=data['filename'], embedding=embedding_str)
                session.add(embedding)

            # Помечаем батч как обработанный
            batch = session.query(Batch).filter_by(id=batch_id).first()
            batch.processed = True

            session.commit()
            logger.info(f"Эмбеддинги для батча {batch_id} успешно сохранены.")
        except Exception as e:
            session.rollback()
            logger.error(f"Ошибка сохранения эмбеддингов для батча {batch_id}: {e}")
            logger.debug(traceback.format_exc())
        finally:
            embeddings_queue.task_done()
        time.sleep(1)  # Добавляем небольшую задержку для снижения нагрузки
    session.close()

# Основная функция скрипта
def main():
    parser = argparse.ArgumentParser(description='Скрипт для парсинга, скачивания и обработки изображений.')
    parser.add_argument('-l', '--limit', type=int, default=1, help='Количество страниц для обработки')
    parser.add_argument('-s', '--start', type=int, default=1, help='Номер стартовой страницы')
    parser.add_argument('-dt', '--download-threads', type=int, default=int(os.environ.get('DOWNLOAD_THREADS', 8)),
                        help='Количество потоков для скачивания изображений в батче (по умолчанию 8)')
    parser.add_argument('-bs', '--batch-size', type=int, default=int(os.environ.get('BATCH_SIZE', 64)),
                        help='Размер батча для обработки изображений (по умолчанию 64)')
    parser.add_argument('-rd', '--report-dir', type=str, default=os.environ.get('REPORT_DIR', 'reports'),
                        help='Директория для сохранения отчётов (по умолчанию "reports")')
    args = parser.parse_args()

    # Чтение переменных окружения
    MACHINE_ID = int(os.environ.get('MACHINE_ID', '0'))
    TOTAL_MACHINES = int(os.environ.get('TOTAL_MACHINES', '1'))
    DOWNLOAD_DIR = os.environ.get('DOWNLOAD_DIR', 'downloads')
    MAX_BATCHES_ON_DISK = int(os.environ.get('MAX_BATCHES_ON_DISK', '5'))
    DOWNLOAD_THREADS = args.download_threads
    BATCH_SIZE = args.batch_size
    REPORT_DIR = args.report_dir
    print(f"MACHINE_ID: {config.MACHINE_ID}, TOTAL_MACHINES: {TOTAL_MACHINES}, DOWNLOAD_DIR: {DOWNLOAD_DIR}, MAX_BATCHES_ON_DISK: {MAX_BATCHES_ON_DISK}, DOWNLOAD_THREADS: {DOWNLOAD_THREADS}, BATCH_SIZE: {BATCH_SIZE}, REPORT_DIR: {REPORT_DIR}")
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    # Настройка базы данных
    engine = get_engine()
    setup_database(engine)
    SessionFactory = get_session_factory(engine)

    # Проверка и удаление не обработанных временных директорий на диске
    session = SessionFactory()
    try:
        batches = session.query(Batch).filter_by(processed=False).all()
        for batch in batches:
            batch_dir = os.path.join(DOWNLOAD_DIR, f"batch_{batch.id}")
            if os.path.exists(batch_dir):
                shutil.rmtree(batch_dir)
                logger.info(f"Удалена не обработанная временная директория: {batch_dir}")
    except Exception as e:
        logger.error(f"Ошибка при очистке временных директорий: {e}")
    finally:
        session.close()

    # Инициализация моделей для извлечения эмбеддингов
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используемое устройство: {device}")

    mtcnn = MTCNN(keep_all=True, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Очереди задач
    page_queue = Queue()
    batch_queue = Queue(maxsize=MAX_BATCHES_ON_DISK)
    embeddings_queue = Queue()

    # Запуск потоков
    downloader = threading.Thread(target=downloader_thread, args=(page_queue, batch_queue, DOWNLOAD_DIR, MAX_BATCHES_ON_DISK, engine, DOWNLOAD_THREADS))
    processor = threading.Thread(target=processing_thread, args=(batch_queue, embeddings_queue, model, mtcnn, device, engine, BATCH_SIZE, REPORT_DIR))
    embeddings_writer = threading.Thread(target=embeddings_writer_thread, args=(embeddings_queue, engine))

    downloader.start()
    processor.start()
    embeddings_writer.start()

    # Заполнение очереди страниц для обработки
    for page_num in range(args.start, args.start + args.limit):
        if (page_num % TOTAL_MACHINES) != MACHINE_ID:
            continue  # Эта страница не для этой машины

        page_url = f"http://camvideos.me/?page={page_num}"
        page_queue.put((page_url, page_num))

    # Завершаем очередь страниц
    page_queue.put(None)
    page_queue.join()

    # Ожидаем, пока все батчи будут обработаны
    batch_queue.join()
    embeddings_queue.join()

    # Завершаем остальные потоки
    batch_queue.put(None)
    embeddings_queue.put(None)

    downloader.join()
    processor.join()
    embeddings_writer.join()

    logger.info("Все батчи обработаны.")

if __name__ == "__main__":
    main()
