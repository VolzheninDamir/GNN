import torch
from simple_model import SimpleHeteroGNN

# Загружаем один граф (укажите правильный путь к файлу, например e1_v00004.pt)
data = torch.load('C:/Users/Damir/Desktop/GNN_dataset_test/GNN_dataset_test/processed/samples/e1_v00001.pt', weights_only=False)

# 1. Замена NaN на 0 (временное решение)
data['cell'].x = torch.nan_to_num(data['cell'].x, nan=0.0)

# 2. Нормализация (как в III_stage_Dataset, но на лету)
cell_mean = data['cell'].x.mean(dim=0, keepdim=True)
cell_std = data['cell'].x.std(dim=0, keepdim=True) + 1e-8
data['cell'].x = (data['cell'].x - cell_mean) / cell_std

# Проверка каждого признака на NaN
for i in range(data['cell'].x.size(1)):
    col = data['cell'].x[:, i]
    if torch.isnan(col).any():
        print(f"Признак {i} содержит NaN: {torch.isnan(col).sum()} из {col.numel()}")

print("cell.x contains NaN:", torch.isnan(data['cell'].x).any())
print("cell.x contains Inf:", torch.isinf(data['cell'].x).any())
print("well.y contains NaN:", torch.isnan(data['well'].y).any())
print("well.y contains Inf:", torch.isinf(data['well'].y).any())
print("well.x contains NaN:", torch.isnan(data['well'].x).any())
print("well.x contains Inf:", torch.isinf(data['well'].x).any())

print("cell.x stats: min={}, max={}, mean={}, std={}".format(
    data['cell'].x.min().item(), data['cell'].x.max().item(),
    data['cell'].x.mean().item(), data['cell'].x.std().item()
))
print("well.x stats: min={}, max={}, mean={}, std={}".format(
    data['well'].x.min().item(), data['well'].x.max().item(),
    data['well'].x.mean().item(), data['well'].x.std().item()
))

# Инициализируем модель
model = SimpleHeteroGNN(
    cell_features=data['cell'].x.size(1),
    well_features=data['well'].x.size(1),
    hidden_dim=64,
    out_seq_len=data['well'].y.size(2),
    num_phases=data['well'].y.size(1)
)

# Прогоняем данные через модель
out = model(data)
print("Выход модели:", out.shape)                     # Ожидается (число_скважин, 3, 25)
print("Целевая переменная:", data['well'].y.shape)    # Должно совпадать

out = model(data)
print("out stats: mean={}, std={}, min={}, max={}".format(
    out.mean().item(), out.std().item(), out.min().item(), out.max().item()
))

# Считаем loss (например, MSE)
loss_fn = torch.nn.MSELoss()
loss = loss_fn(out, data['well'].y)
print("Потеря на одном графе:", loss.item())