from keras_applications.imagenet_utils import _obtain_input_shape
from keras_vggface.vggface import VGGFace

# Convolution Features
vgg_features = VGGFace(include_top=False, input_shape=(160, 160, 3), pooling='avg')
model = vgg_features
