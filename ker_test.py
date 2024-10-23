from keras_facenet import FaceNet

# Загрузка модели FaceNet, предобученной на базе InceptionResNetV1
embedder = FaceNet()

# Получение модели
model = embedder.model

# Веса загружаются автоматически, поэтому дополнительная загрузка не требуется

# Пример получения эмбеддингов для изображения
def get_embeddings(image):
    embeddings = embedder.embeddings([image])
    return embeddings

# Пример использования модели с изображением
import numpy as np
from PIL import Image

# Загрузка изображения и преобразование его в numpy array
image = Image.open('bae_cake__07102024_1845_female_Chaturbate.jpg').resize((160, 160))
image_np = np.array(image)

# Получение эмбеддингов
embeddings = get_embeddings(image_np)
print(embeddings)

