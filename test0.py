from facenet_pytorch import InceptionResnetV1
import torch

# Загрузка предобученной модели (предобученная на VGGFace2)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Пример использования модели для извлечения эмбеддингов лица
def get_embeddings(face_image):
    face_image = face_image.permute(2, 0, 1).unsqueeze(0)  # Преобразование в (1, 3, 160, 160)
    embeddings = model(face_image)
    return embeddings

# Пример вызова
embeddings = get_embeddings(your_face_image_tensor)
print(embeddings)

