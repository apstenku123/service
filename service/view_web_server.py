# file: view_web_server.py
# directory: .
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
from utils import preprocess_image
import logging
import os
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sqlalchemy.orm import sessionmaker, scoped_session
from models import ImageEmbedding, Image, BaseImageUrl, ArchivedImage  # Убедитесь, что ArchivedImage импортирован
from sqlalchemy import create_engine, text, inspect

# Настройки приложения
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Настройки логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация моделей для эмбеддингов
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info(f"Используется устройство: {device}")

mtcnn = MTCNN(keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Подключение к базе данных
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_NAME = os.environ.get('DB_NAME', 'mydatabase')
DB_USER = os.environ.get('DB_USER', 'myuser')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'mypassword')

DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
engine = create_engine(DATABASE_URL)
SessionFactory = sessionmaker(bind=engine)
Session = scoped_session(SessionFactory)

# Инспектор для проверки наличия таблиц
inspector = inspect(engine)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # Обработка загруженного изображения
            try:
                img_rgb = preprocess_image(upload_path)
                if img_rgb is None:
                    logger.error("Не удалось загрузить изображение.")
                    return render_template('index.html', message="Не удалось загрузить изображение.")

                # Детектирование лица и получение эмбеддинга
                face = mtcnn(img_rgb)
                if face is None:
                    logger.info("На загруженном изображении не обнаружено лицо.")
                    return render_template('index.html', message="На загруженном изображении не обнаружено лицо.")

                face = face.unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = model(face).cpu().numpy()

                # Поиск похожих эмбеддингов в базе данных
                embedding_vector = embedding.flatten().tolist()  # Преобразуем в список чисел

                # Создаем строковое представление эмбеддинга для использования в SQL
                embedding_list_str = ','.join(map(str, embedding_vector))
                embedding_sql = f'ARRAY[{embedding_list_str}]'

                with Session() as session:
                    # Проверяем, есть ли эмбеддинги в базе данных
                    count_embeddings = session.query(ImageEmbedding).count()
                    if count_embeddings == 0:
                        logger.info("В базе данных нет эмбеддингов.")
                        return render_template('index.html', message="В базе данных нет эмбеддингов.")

                    # Выполняем обновленный SQL-запрос
                    sql_query = f"""
                        WITH ranked_embeddings AS (
                            SELECT
                                image_id,
                                filename,
                                embedding,
                                embedding <=> {embedding_sql}::vector AS distance,
                                ROW_NUMBER() OVER (
                                    PARTITION BY image_id
                                    ORDER BY embedding <=> {embedding_sql}::vector
                                ) AS rn
                            FROM
                                image_embeddings
                        )
                        SELECT
                            image_id,
                            filename
                        FROM
                            ranked_embeddings
                        WHERE
                            rn = 1
                        ORDER BY
                            distance
                        LIMIT 10;
                    """

                    results = session.execute(text(sql_query)).mappings().all()

                    # Получение URL изображений
                    similar_images = []
                    for row in results:
                        image_id = row['image_id']
                        filename = row['filename']

                        # Проверяем наличие таблицы ArchivedImage
                        if 'archived_images' in inspector.get_table_names():
                            # Попробуем получить архивированное изображение
                            try:
                                archived_image = session.query(ArchivedImage).filter_by(image_id=image_id).first()
                                if archived_image:
                                    image_url = archived_image.archive_url
                                else:
                                    # Используем оригинальный URL
                                    image = session.query(Image).filter_by(id=image_id).first()
                                    if image is None:
                                        logger.error(f"Изображение с id {image_id} не найдено.")
                                        continue

                                    base_image_url = session.query(BaseImageUrl).filter_by(id=image.base_url_id).first()
                                    if base_image_url is None:
                                        logger.error(f"Базовый URL с id {image.base_url_id} не найден.")
                                        continue

                                    base_url = base_image_url.base_url
                                    image_url = f"{base_url}/{filename}"
                            except Exception as e:
                                logger.error(f"Ошибка при обращении к таблице archived_images: {e}")
                                # Используем оригинальный URL
                                image = session.query(Image).filter_by(id=image_id).first()
                                if image is None:
                                    logger.error(f"Изображение с id {image_id} не найдено.")
                                    continue

                                base_image_url = session.query(BaseImageUrl).filter_by(id=image.base_url_id).first()
                                if base_image_url is None:
                                    logger.error(f"Базовый URL с id {image.base_url_id} не найден.")
                                    continue

                                base_url = base_image_url.base_url
                                image_url = f"{base_url}/{filename}"
                        else:
                            # Таблица archived_images не существует, используем оригинальный URL
                            image = session.query(Image).filter_by(id=image_id).first()
                            if image is None:
                                logger.error(f"Изображение с id {image_id} не найдено.")
                                continue

                            base_image_url = session.query(BaseImageUrl).filter_by(id=image.base_url_id).first()
                            if base_image_url is None:
                                logger.error(f"Базовый URL с id {image.base_url_id} не найден.")
                                continue

                            base_url = base_image_url.base_url
                            image_url = f"{base_url}/{filename}"

                        similar_images.append(image_url)

                # Возвращаем результат после цикла
                return render_template('results.html', images=similar_images)
            except Exception as e:
                logger.error(f"Ошибка при обработке загруженного изображения: {e}", exc_info=True)
                return render_template('index.html', message="Произошла ошибка при обработке изображения.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8070, debug=True)
