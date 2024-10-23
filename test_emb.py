import os
import cv2
import torch
import base64
import numpy as np
from facenet_pytorch import InceptionResnetV1
from insightface.app import FaceAnalysis
import argparse

# Функция для создания временной директории для каждого изображения
def create_tmp_directory(image_path):
    base_name = os.path.basename(image_path).split('.')[0]  # Получаем имя файла без расширения
    tmp_dir = f"tmp/{base_name}_tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    return tmp_dir

# Сохранение лица как изображения в указанную директорию
def save_face(face, model_name, tmp_dir, face_idx):
    file_path = f"{tmp_dir}/{face_idx}_{model_name}.jpg"
    cv2.imwrite(file_path, face)
    return file_path

# Конвертация изображения в Base64
def convert_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Преобразование лица по боксу
def crop_face(img, box):
    x, y, w, h = box
    face = img[y:y + h, x:x + w]
    return face

# Преобразование изображения в нужный формат для модели
def preprocess_face_for_embedding(face, target_size=(160, 160)):
    face = cv2.resize(face, target_size)  # Изменение размера
    face = face.astype('float32') / 255.0  # Нормализация
    return face

# Модели для извлечения эмбеддингов
arcface_model = torch.jit.load('backbone.pth')
app = FaceAnalysis(name='antelopev2')
app.prepare(ctx_id=0, nms=0.4)

# Загрузка модели InceptionResNetV1 для ArcFace (из facenet-pytorch)
inception_resnet_v1 = InceptionResnetV1(pretrained='vggface2').eval()

# Извлечение эмбеддингов для каждой модели
def get_inception_resnet_embeddings(face):
    face = preprocess_face_for_embedding(face, target_size=(160, 160))
    face_tensor = torch.tensor(face).permute(2, 0, 1).unsqueeze(0)
    face_tensor = face_tensor.to(torch.float32)
    with torch.no_grad():
        embedding = inception_resnet_v1(face_tensor).detach().cpu().numpy()
    return embedding

def get_arcface_embeddings(face):
    face = preprocess_face_for_embedding(face, target_size=(112, 112))
    face_tensor = torch.tensor(face).permute(2, 0, 1).unsqueeze(0)
    face_tensor = face_tensor.to(torch.float32)
    return arcface_model(face_tensor).detach().cpu().numpy()

def get_insightface_embeddings(face):
    face = preprocess_face_for_embedding(face, target_size=(112, 112))
    return app.get(face)[0]['embedding']

# Функция для извлечения эмбеддингов и сохранения лиц
def extract_face_embeddings(image_path):
    # Загружаем изображение
    img = cv2.imread(image_path)
    
    # Создаем временную директорию для хранения лиц
    tmp_dir = create_tmp_directory(image_path)

    # Используем каскадную функцию для получения боксов лиц
    faces = cascade_face_detection(image_path)

    if faces:
        embeddings = {
            'inception_resnet_v1': [],
            'arcface': [],
            'insightface': []
        }
        face_data_for_db = []
        for idx, face_box in enumerate(faces):
            face = crop_face(img, face_box)  # Выделяем лицо по боксу

            # 1. Получаем эмбеддинги через InceptionResNetV1
            inception_resnet_embedding = get_inception_resnet_embeddings(face)
            embeddings['inception_resnet_v1'].append(inception_resnet_embedding)
            inception_resnet_face_path = save_face(face, 'inception_resnet_v1', tmp_dir, idx)
            inception_resnet_face_base64 = convert_to_base64(inception_resnet_face_path)
            face_data_for_db.append({
                'model': 'inception_resnet_v1',
                'embedding': inception_resnet_embedding,
                'image_base64': inception_resnet_face_base64
            })

            # 2. Получаем эмбеддинги через ArcFace
            arcface_embedding = get_arcface_embeddings(face)
            embeddings['arcface'].append(arcface_embedding)
            arcface_face_path = save_face(face, 'arcface', tmp_dir, idx)
            arcface_face_base64 = convert_to_base64(arcface_face_path)
            face_data_for_db.append({
                'model': 'arcface',
                'embedding': arcface_embedding,
                'image_base64': arcface_face_base64
            })

            # 3. Получаем эмбеддинги через InsightFace
            insightface_embedding = get_insightface_embeddings(face)
            embeddings['insightface'].append(insightface_embedding)
            insightface_face_path = save_face(face, 'insightface', tmp_dir, idx)
            insightface_face_base64 = convert_to_base64(insightface_face_path)
            face_data_for_db.append({
                'model': 'insightface',
                'embedding': insightface_embedding,
                'image_base64': insightface_face_base64
            })

        return face_data_for_db  # Возвращаем данные для записи в базу
    else:
        print("Лицо не найдено.")
        return None

# Обработка аргументов командной строки
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Извлечение лицевых эмбеддингов из изображения')
    parser.add_argument('--image', required=True, help='Путь к изображению')
    args = parser.parse_args()

    # Вызов функции для обработки изображения
    result = extract_face_embeddings(args.image)
    
    if result:
        print("Лица и эмбеддинги успешно извлечены и сохранены.")
    else:
        print("Лицо не найдено.")

