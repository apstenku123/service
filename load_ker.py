
from keras_facenet import FaceNet

# Создаем модель Inception ResNet v1 (facenet)
embedder = FaceNet()

#Step 2
# Step 2: Доступ к модели InceptionResNetV1 внутри
model = embedder.model
model.load_weights('keras-facenet-h5/facenet_keras_weights.h5')
