import torch
import torch.nn as nn
from simple_model import SimpleHeteroGNN

# Загрузка данных
path = 'C:/Users/Damir/Desktop/GNN_dataset_test/GNN_dataset_test/processed/samples/e1_v00001.pt'
data = torch.load(path, weights_only=False)

# --- НОРМАЛИЗАЦИЯ ---

# 1. Чистим NaN
data['cell'].x = torch.nan_to_num(data['cell'].x, nan=0.0)

# 2. Нормализуем входные ячейки (Z-score)
c_mean, c_std = data['cell'].x.mean(dim=0), data['cell'].x.std(dim=0) + 1e-8
data['cell'].x = (data['cell'].x - c_mean) / c_std

# 3. !!! НОРМАЛИЗУЕМ ТАРГЕТ (well.y) !!!
# Это критически важно для уменьшения лосса
y_min = data['well'].y.min()
y_max = data['well'].y.max()
# Сохраним оригинал для проверки, а в данных заменим на 0-1
target_normalized = (data['well'].y - y_min) / (y_max - y_min + 1e-8)

# --- ЗАПУСК МОДЕЛИ ---

model = SimpleHeteroGNN(
    cell_features=data['cell'].x.size(1),
    well_features=data['well'].x.size(1),
    hidden_dim=64
)

# Предсказание
out = model(data)

# Считаем лосс на нормализованных данных
loss_fn = torch.nn.MSELoss()
loss = loss_fn(out, target_normalized)

print(f"Shape out: {out.shape}")
print(f"Loss (normalized): {loss.item():.6f}")

# Чтобы увидеть реальную ошибку в кубах, нужно денормализовать:
out_real = out * (y_max - y_min) + y_min
loss_real = torch.sqrt(loss_fn(out_real, data['well'].y)) # RMSE в реальных единицах
print(f"Средняя ошибка в реальных единицах (кубах/тоннах): {loss_real.item():.2f}")