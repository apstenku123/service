# file: test22.py
# directory: test_nv
import argparse
import os
import torch
import cv2
import numpy as np

from facenet_pytorch import MTCNN, InceptionResnetV1
from insightface.app import FaceAnalysis
from insightface.utils import face_align

def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Face Embedding Tester')
    parser.add_argument('-s', '--image_file', type=str, required=True, help='Путь к файлу изображения')
    args = parser.parse_args()

    image_file = args.image_file

    if not os.path.exists(image_file):
        print(f"Файл изображения {image_file} не существует.")
        return

    # Инициализация устройства (CUDA, если доступно)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используемое устройство: {device}')

    # Инициализация MTCNN с keep_all=True
    mtcnn = MTCNN(keep_all=True, device=device)

    # Инициализация модели InceptionResnetV1
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Инициализация модели InsightFace
    app = FaceAnalysis(allowed_modules=['detection', 'recognition'],
                       providers=['CUDAExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider'])
    app.prepare(ctx_id=0 if device.type == 'cuda' else -1, det_size=(640, 640))

    # Получаем модель распознавания
    recognizer = app.models['recognition']

    # Чтение изображения
    img_bgr = cv2.imread(image_file)
    if img_bgr is None:
        print(f"Не удалось загрузить изображение: {image_file}")
        return

    # Конвертация изображения в RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Обнаружение лиц с помощью MTCNN и получение координат лиц и ключевых точек
    boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)

    if boxes is not None and landmarks is not None:
        num_faces = boxes.shape[0]
        print(f'Количество лиц, обнаруженных MTCNN: {num_faces}')

        # Получение выровненных лиц для InceptionResnetV1
        aligned_faces = mtcnn(img_rgb)  # Возвращает тензор с выровненными лицами
        if isinstance(aligned_faces, torch.Tensor):
            face_tensors = aligned_faces.to(device)
        else:
            print("Не удалось получить выровненные лица для InceptionResnetV1.")
            return

        # Получение эмбеддингов от InceptionResnetV1
        with torch.no_grad():
            embeddings_facenet = facenet_model(face_tensors).cpu().numpy()

        num_embeddings_facenet = embeddings_facenet.shape[0]
        print(f'Количество эмбеддингов от InceptionResnetV1: {num_embeddings_facenet}')

        # Получение эмбеддингов от InsightFace
        embeddings_insightface = []
        for i in range(num_faces):
            # landmarks[i] имеет форму (5, 2)
            landmark = landmarks[i]

            # Преобразуем landmark в numpy.array с типом float32
            landmark = np.array(landmark, dtype=np.float32)
            if landmark.shape != (5, 2):
                print(f"Неверная форма landmark для лица {i}: {landmark.shape}. Пропускаем.")
                continue

            if np.any(np.isnan(landmark)) or np.any(np.isinf(landmark)):
                print(f"Invalid landmark values for face {i}. Skipping.")
                continue

            # Используем face_align.norm_crop для выравнивания лица
            face_aligned = face_align.norm_crop(img_rgb, landmark=landmark, image_size=112)
            # face_aligned имеет размер (112, 112, 3) и находится в формате RGB

            # Конвертируем изображение в BGR, если это необходимо
            face_aligned_bgr = cv2.cvtColor(face_aligned, cv2.COLOR_RGB2BGR)

            # Убеждаемся, что изображение имеет тип uint8
            face_aligned_bgr = face_aligned_bgr.astype(np.uint8)

            # Получение эмбеддинга с помощью recognizer
            embedding_insight = recognizer.get_feat([face_aligned_bgr])
            embeddings_insightface.append(embedding_insight)

        num_embeddings_insightface = len(embeddings_insightface)
        print(f'Количество эмбеддингов от InsightFace: {num_embeddings_insightface}')

    else:
        print('MTCNN не обнаружил лиц на изображении.')

if __name__ == '__main__':
    main()
