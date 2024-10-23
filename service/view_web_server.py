# file: view_web_server.py
# directory: .
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
from PIL import Image as PILImage, UnidentifiedImageError
import pyheif
import logging
import os
import torch
import numpy as np
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from sqlalchemy.orm import sessionmaker, scoped_session
from models import ImageEmbedding, Image, BaseImageUrl, ArchivedImage
from sqlalchemy import create_engine, text

# Настройки приложения
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FAVICON_FOLDER'] = 'favicon'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FAVICON_FOLDER'], exist_ok=True)
# Настройки логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Поддерживаемые форматы изображений
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'heic', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_rgb(image_path):
    try:
        extension = image_path.rsplit('.', 1)[1].lower()
        if extension == 'heic':
            heif_file = pyheif.read(image_path)
            img = PILImage.frombytes(
                heif_file.mode, heif_file.size, heif_file.data,
                "raw", heif_file.mode, heif_file.stride
            )
        else:
            img = PILImage.open(image_path)
        img = img.convert('RGB')
        return np.array(img)
    except (UnidentifiedImageError, ValueError, pyheif.error.HeifError) as e:
        logger.error(f"Не удалось определить формат изображения: {image_path}, ошибка: {e}")
        return None

# Инициализация моделей для эмбеддингов
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info(f"Используется устройство: {device}")

# Инициализируем MTCNN с keep_all=True для обнаружения всех лиц
mtcnn = MTCNN(keep_all=True, device=device)
# Инициализируем модель InceptionResnetV1 для получения эмбеддингов FaceNet
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
# Инициализируем модель InsightFace для получения эмбеддингов InsightFace
app_insight = FaceAnalysis(allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider'])
app_insight.prepare(ctx_id=0 if device.type == 'cuda' else -1, det_size=(640, 640))

# Подключение к базе данных
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_NAME = os.environ.get('DB_NAME', 'mydatabase')
DB_USER = os.environ.get('DB_USER', 'myuser')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'mypassword')

DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
engine = create_engine(DATABASE_URL)
SessionFactory = sessionmaker(bind=engine)
Session = scoped_session(SessionFactory)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.config['FAVICON_FOLDER'], 'favicon.ico', mimetype='image/vnd.microsoft.icon')
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # Конвертируем изображение в RGB
            img_rgb = convert_to_rgb(upload_path)
            if img_rgb is None:
                return render_template('index.html', message="Не удалось загрузить изображение.")

            # Обработка загруженного изображения
            try:
                # Детектирование лиц и получение координат и ключевых точек с помощью MTCNN
                boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)
                if boxes is None or len(boxes) == 0:
                    logger.info("На загруженном изображении не обнаружено лицо.")
                    return render_template('index.html', message="На загруженном изображении не обнаружено лицо.")

                # Используем первое обнаруженное лицо
                box = boxes[0]
                landmark = landmarks[0]

                # Преобразуем landmark в numpy-массив с типом float64
                landmark = np.array(landmark, dtype=np.float64)

                # Извлекаем лицо для InceptionResnetV1
                face_aligned = mtcnn.extract(img_rgb, [box], save_path=None)
                if face_aligned is None or len(face_aligned) == 0:
                    logger.error("Не удалось извлечь лицо.")
                    return render_template('index.html', message="Не удалось извлечь лицо.")

                face_tensor = face_aligned[0].unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding_facenet = facenet_model(face_tensor).cpu().numpy()

                # Выравниваем лицо для InsightFace
                face_aligned_insight = face_align.norm_crop(img_rgb, landmark=landmark, image_size=112)

                # Конвертируем в BGR и uint8
                face_aligned_insight_bgr = cv2.cvtColor(face_aligned_insight, cv2.COLOR_RGB2BGR)
                face_aligned_insight_bgr = face_aligned_insight_bgr.astype('uint8')

                # Получаем эмбеддинг с помощью InsightFace
                embedding_insight = app_insight.models['recognition'].get_feat(face_aligned_insight_bgr)

                # Подготавливаем эмбеддинги для SQL-запроса
                embedding_facenet_vector = embedding_facenet.flatten()
                embedding_insight_vector = embedding_insight.flatten()

                embedding_facenet_vector_list = embedding_facenet_vector.tolist()
                embedding_insight_vector_list = embedding_insight_vector.tolist()

                embedding_facenet_list_str = ','.join(map(str, embedding_facenet_vector_list))
                embedding_facenet_sql = f'ARRAY[{embedding_facenet_list_str}]'

                embedding_insight_list_str = ','.join(map(str, embedding_insight_vector_list))
                embedding_insight_sql = f'ARRAY[{embedding_insight_list_str}]'

                with Session() as session:
                    # Проверяем, есть ли эмбеддинги в базе данных
                    count_embeddings = session.query(ImageEmbedding).count()
                    if count_embeddings == 0:
                        logger.info("В базе данных нет эмбеддингов.")
                        return render_template('index.html', message="В базе данных нет эмбеддингов.")

                # Выполняем SQL-запрос, используя оба эмбеддинга и усредняя расстояния
                sql_query = f"""
                    WITH distances AS (
                        SELECT
                            ie.image_id,
                            ie.filename,
                            COALESCE((ie.embedding <=> {embedding_facenet_sql}::vector), 2.0) AS facenet_distance,
                            COALESCE((ie.insightface_embedding <=> {embedding_insight_sql}::vector), 2.0) AS insightface_distance,
                            ((COALESCE((ie.embedding <=> {embedding_facenet_sql}::vector), 2.0) + COALESCE((ie.insightface_embedding <=> {embedding_insight_sql}::vector), 2.0)) / 2) AS avg_distance
                        FROM
                            image_embeddings ie
                    )
                    SELECT
                        d.image_id,
                        d.filename,
                        i.path,
                        d.facenet_distance,
                        d.insightface_distance,
                        d.avg_distance,
                        ai.archive_url,
                        biu.base_url
                    FROM
                        distances d
                    LEFT JOIN images i ON d.image_id = i.id
                    LEFT JOIN base_image_urls biu ON i.base_url_id = biu.id
                    LEFT JOIN archived_images ai ON d.image_id = ai.image_id
                    ORDER BY
                        d.avg_distance
                    LIMIT 10;
                """

                results = session.execute(text(sql_query)).mappings().all()

                # Получение URL изображений
                similar_images = []
                for row in results:
                    image_id = row['image_id']
                    filename = row['filename']
                    path = row['path']
                    archive_url = row['archive_url']
                    base_url = row['base_url']
                    facenet_distance = row['facenet_distance']
                    insightface_distance = row['insightface_distance']
                    avg_distance = row['avg_distance']

                    # Предпочитаем archive_url, если он есть
                    if archive_url:
                        image_url = archive_url
                    elif base_url:
                        if path:
                            image_url = f"{base_url}/{path}/{filename}"
                        else:
                            image_url = f"{base_url}/{filename}"
                    else:
                        logger.error(f"Не удалось получить URL для изображения с id {image_id}.")
                        continue

                    similar_images.append({
                        'image_url': image_url,
                        'facenet_distance': facenet_distance,
                        'insightface_distance': insightface_distance,
                        'avg_distance': avg_distance
                    })

                # Возвращаем результат
                return render_template('results.html', images=similar_images)
            except Exception as e:
                logger.error(f"Ошибка при обработке загруженного изображения: {e}", exc_info=True)
                return render_template('index.html', message="Произошла ошибка при обработке изображения.")
    return render_template('index.html')

if __name__ == '__main__':
    # Запуск приложения без перезагрузчика
    app.run(host='0.0.0.0', port=8070, debug=True, use_reloader=False)
