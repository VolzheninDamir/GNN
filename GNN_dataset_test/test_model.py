import torch
import torch.nn as nn
import numpy as np  # добавили импорт
from simple_model import SimpleHeteroGNN

# Загрузка данных
path = 'C:/Users/Damir/Desktop/GNN_dataset_test/GNN_dataset_test/processed/samples/e1_v00001.pt'
data = torch.load(path, weights_only=False)

# --- Преобразование edge_attr в тензор ---
if ('cell', 'flows_to', 'cell') in data.edge_types:
    edge_attr = data['cell', 'flows_to', 'cell'].get('edge_attr', None)
    if edge_attr is not None and isinstance(edge_attr, np.ndarray):
        data['cell', 'flows_to', 'cell'].edge_attr = torch.from_numpy(edge_attr).float()

# --- НОРМАЛИЗАЦИЯ (ручная, для одиночного теста) ---
data['cell'].x = torch.nan_to_num(data['cell'].x, nan=0.0)
c_mean, c_std = data['cell'].x.mean(dim=0), data['cell'].x.std(dim=0) + 1e-8
data['cell'].x = (data['cell'].x - c_mean) / c_std

y_min = data['well'].y.min()
y_max = data['well'].y.max()
target_normalized = (data['well'].y - y_min) / (y_max - y_min + 1e-8)

# --- Определим размерность edge_attr ---
edge_dim = 1
if ('cell', 'flows_to', 'cell') in data.edge_types:
    edge_attr = data['cell', 'flows_to', 'cell'].get('edge_attr', None)
    if edge_attr is not None:
        if isinstance(edge_attr, torch.Tensor):
            edge_dim = edge_attr.size(1) if edge_attr.dim() == 2 else 1
        elif isinstance(edge_attr, np.ndarray):
            edge_dim = edge_attr.shape[1] if edge_attr.ndim == 2 else 1

# --- ЗАПУСК МОДЕЛИ ---
model = SimpleHeteroGNN(
    cell_features=data['cell'].x.size(1),
    well_features=data['well'].x.size(1),
    hidden_dim=64,
    edge_dim=edge_dim
)

out = model(data)
loss_fn = torch.nn.MSELoss()
loss = loss_fn(out, target_normalized)

print(f"Shape out: {out.shape}")
print(f"Loss (normalized): {loss.item():.6f}")

out_real = out * (y_max - y_min) + y_min
loss_real = torch.sqrt(loss_fn(out_real, data['well'].y))
print(f"Средняя ошибка в реальных единицах (кубах/тоннах): {loss_real.item():.2f}")