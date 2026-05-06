import torch
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.loader import DataLoader
import torch.nn as nn

# Импортируем новую модель и функции загрузки
from gnn_plus_hetero_v2 import GNNPlusHetero
from utils import load_graph_data

class WeightedMSELoss(nn.Module):
    def __init__(self, num_steps=24):
        super().__init__()
        # Линейный рост от 0.2 (шаг 2) до 1.0 (последний шаг)
        t = torch.linspace(0, 1, num_steps)        # [0.0, 0.0435, ..., 1.0]
        weights = 0.2 + 0.8 * t                    # от 0.2 до 1.0
        # Корректируем первые шаги
        weights[0] = 1.0    # первый шаг вес 1.0
        weights[1] = 0.2    # второй шаг 0.2
        self.register_buffer('weights', weights)

    def forward(self, pred, target):
        diff = pred - target
        w = self.weights.to(pred.device)
        loss = (diff ** 2) * w.view(1, 1, -1)
        return loss.mean()

def train_model():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Работаем на: {device}")

    # Загрузка данных
    train_loader, val_loader, feat_list = load_graph_data(
        config['paths'],
        config['preprocess'],
        config['train']
    )

    # Получаем образец для определения размерностей
    sample_data, _ = next(iter(train_loader))

    # Размерность признаков рёбер cell→cell
    edge_dim = 1
    if ('cell', 'flows_to', 'cell') in sample_data.edge_types:
        edge_attr = sample_data['cell', 'flows_to', 'cell'].get('edge_attr', None)
        if edge_attr is not None:
            if isinstance(edge_attr, torch.Tensor):
                edge_dim = edge_attr.size(1) if edge_attr.dim() == 2 else 1
            elif isinstance(edge_attr, np.ndarray):
                edge_dim = edge_attr.shape[1] if edge_attr.ndim == 2 else 1

    print(f"Размерность edge_attr: {edge_dim}")

    # Инициализация новой модели
    model = GNNPlusHetero(
        cell_features=sample_data['cell'].x.size(1),
        well_features=sample_data['well'].x.size(1),
        hidden_dim=config['model']['nz'],
        out_seq_len=24,
        num_phases=3,
        edge_dim=edge_dim,
        num_layers=config['model'].get('num_layers', 3),      # можно добавить в yaml
        dropout=config['model'].get('dropout', 0.1),
        # use_lstm=config['model'].get('use_lstm', False)
    ).to(device)

    # Оптимизатор и функция потерь
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['train']['warmup_learning_rate'],
        weight_decay=config['train']['weight_decay']
    )
    
    num_steps = 24   # длина временного ряда (out_seq_len)
    criterion = WeightedMSELoss(num_steps=num_steps)

    epochs = config['train']['epochs']
    train_history, val_history = [], []

    print(f"Начинаем обучение на {len(train_loader.dataset)} графах...")

    for epoch in range(1, epochs + 1):
        # --- Тренировка ---
        model.train()
        total_train_loss = 0
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch)
            loss = criterion(pred, batch['well'].y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_history.append(avg_train_loss)

        # --- Валидация ---
        model.eval()
        total_val_loss = 0
        total_val_mse = 0           # добавим накопитель обычной MSE
        with torch.no_grad():
            for v_batch, _ in val_loader:
                v_batch = v_batch.to(device)
                v_pred = model(v_batch)
                v_loss = criterion(v_pred, v_batch['well'].y)   # взвешенный
                v_mse = nn.MSELoss()(v_pred, v_batch['well'].y)  # обычный
                total_val_loss += v_loss.item()
                total_val_mse += v_mse.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_mse = total_val_mse / len(val_loader)
        val_history.append(avg_val_loss)   # для графика взвешенного loss
        # можно также сохранять avg_val_mse в отдельный список и построить его график

        if epoch % 1 == 0:
            print(f"Эпоха {epoch:03d} | Train MSE: {avg_train_loss:.6f} | Val MSE (обычный): {avg_val_mse:.6f} | Val Loss (взв): {avg_val_loss:.6f}")

    # Сохранение модели
    os.makedirs(config['paths']['models'], exist_ok=True)
    save_path = os.path.join(config['paths']['models'], 'gnn_plus_oil_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Модель сохранена в {save_path}")

    # График лосса
    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label='Train Loss')
    plt.plot(val_history, label='Val Loss')
    plt.yscale('log')
    plt.xlabel('Эпоха')
    plt.ylabel('MSE (нормализованный)')
    plt.legend()
    plt.title('Процесс обучения GNN+')
    plt.show()

if __name__ == "__main__":
    train_model()