import torch

# Загружаем веса из файла backbone.pth
weights = torch.load('backbone.pth')

# Выводим ключи состояния модели, чтобы посмотреть, какие слои сохранены
print(weights.keys())

# Для подробного просмотра можно напечатать форму каждого из сохранённых тензоров
for key, value in weights.items():
    print(f"{key}: {value.shape}")

