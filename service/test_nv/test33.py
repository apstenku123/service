# file: test33.py
# directory: test_nv

import argparse
import os
import torch
import cv2
import sys

import cupy as cp

from facenet_pytorch import MTCNN, InceptionResnetV1
from insightface.app import FaceAnalysis
from insightface.utils import face_align

def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Face Embedding Tester')
    parser.add_argument('-s', '--image_file', type=str, help='Путь к файлу изображения')
    parser.add_argument('-d', '--directory', type=str, help='Путь к директории с изображениями')
    args = parser.parse_args()

    if not args.image_file and not args.directory:
        print("Пожалуйста, укажите файл изображения с помощью -s или директорию с помощью -d.")
        sys.exit(1)

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

    # Счетчик для очистки кеша GPU
    images_processed = 0

    # Функция для обработки одного изображения
    def process_image(image_path):
        nonlocal images_processed

        # Чтение изображения с помощью OpenCV с поддержкой CUDA
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            return

        # Конвертация изображения в RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Обнаружение лиц с помощью MTCNN и получение координат лиц и ключевых точек
        boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)

        if boxes is not None and landmarks is not None:
            num_faces = boxes.shape[0]
            print(f'Изображение: {os.path.basename(image_path)} - Количество лиц, обнаруженных MTCNN: {num_faces}')

            # Получение выровненных лиц для InceptionResnetV1
            aligned_faces = mtcnn(img_rgb)  # Возвращает тензор с выровненными лицами
            if isinstance(aligned_faces, torch.Tensor):
                # Перенос лиц на устройство с использованием pin_memory и non_blocking
                face_tensors = [face.pin_memory().to(device, non_blocking=True) for face in aligned_faces]

                # Объединение тензоров лиц
                face_tensors = torch.stack(face_tensors)
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

                # Преобразуем landmark в cupy.array с типом float32
                landmark_cp = cp.asarray(landmark, dtype=cp.float32)
                if landmark_cp.shape != (5, 2):
                    print(f"Неверная форма landmark для лица {i}: {landmark_cp.shape}. Пропускаем.")
                    continue

                if cp.isnan(landmark_cp).any() or cp.isinf(landmark_cp).any():
                    print(f"Недопустимые значения landmark для лица {i}. Пропускаем.")
                    continue

                # Конвертируем landmark обратно в host memory для face_align
                landmark_np = cp.asnumpy(landmark_cp)

                # Используем face_align.norm_crop для выравнивания лица
                face_aligned = face_align.norm_crop(img_rgb, landmark=landmark_np, image_size=112)
                # face_aligned имеет размер (112, 112, 3) и находится в формате RGB

                # Конвертируем изображение в BGR
                face_aligned_bgr = cv2.cvtColor(face_aligned, cv2.COLOR_RGB2BGR)

                # Убеждаемся, что изображение имеет тип uint8 и находится в GPU памяти
                face_aligned_bgr_cp = cp.asarray(face_aligned_bgr.astype('uint8'))

                # Получение эмбеддинга с помощью recognizer
                embedding_insight = recognizer.get_feat([cp.asnumpy(face_aligned_bgr_cp)])
                embeddings_insightface.append(embedding_insight)

            num_embeddings_insightface = len(embeddings_insightface)
            print(f'Количество эмбеддингов от InsightFace: {num_embeddings_insightface}')

        else:
            print(f'MTCNN не обнаружил лиц на изображении: {os.path.basename(image_path)}')

        images_processed += 1

        # Очистка кеша GPU каждые 200 изображений
        if images_processed % 200 == 0:
            torch.cuda.empty_cache()
            print("Кеш GPU очищен.")

    if args.image_file:
        if not os.path.exists(args.image_file):
            print(f"Файл изображения {args.image_file} не существует.")
            sys.exit(1)
        process_image(args.image_file)

    elif args.directory:
        if not os.path.exists(args.directory):
            print(f"Директория {args.directory} не существует.")
            sys.exit(1)

        # Получаем список файлов изображений в директории
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [os.path.join(args.directory, f) for f in os.listdir(args.directory)
                       if f.lower().endswith(supported_formats)]

        if not image_files:
            print(f"В директории {args.directory} не найдено файлов изображений.")
            sys.exit(1)

        print(f"Обработка {len(image_files)} изображений в директории {args.directory}.")

        for image_path in image_files:
            process_image(image_path)

    print("Обработка завершена.")

if __name__ == '__main__':
    main()
