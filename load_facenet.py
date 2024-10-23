from keras.models import load_model
import tensorflow as tf

# Загружаем модель FaceNet
facenet_model = load_model('facenet_keras.h5')

# Проверьте, если модель корректно загружена
print(facenet_model.summary())
