# file: check_cv_cuda.py
# directory: test_nv
import cv2

# Проверьте наличие модуля CUDA в OpenCV
cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0

if cuda_available:
     print("OpenCV поддерживает CUDA и обнаружен GPU.")
     print(f"Количество доступных CUDA устройств: {cv2.cuda.getCudaEnabledDeviceCount()}")
else:
     print("OpenCV не поддерживает CUDA или не обнаружено доступных устройств CUDA.")


import os

# Узнать версию OpenCV
print(f"Версия OpenCV: {cv2.__version__}")

# Узнать путь к файлу библиотеки OpenCV
opencv_path = os.path.abspath(cv2.__file__)
print(f"Путь к файлу библиотеки OpenCV: {opencv_path}")

opencv_shared_object = os.path.join(os.path.dirname(opencv_path), 'cv2.so')
print(f"Путь к файлу .so библиотеки OpenCV: {opencv_shared_object}")

# print(cv2.getBuildInformation())
