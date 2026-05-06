import os
import torch
import yaml
import optuna
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GATv2Conv, Linear, TransformerConv, HeteroConv
import torch.nn as nn
import torch.nn.functional as F

# Импорт функции загрузки данных из вашего модуля
from utils import load_graph_data


# ----------------------------------------------------------------------
# Гибкий слой GNN+ с выбором типа свёртки (поддержка edge_attr)
# ----------------------------------------------------------------------
class GNNPlusLayerFlex(nn.Module):
    """
    Один блок архитектуры GNN+, где тип message passing слоя выбирается динамически.
    Поддерживаемые типы:
        - 'GATConv'
        - 'GATv2Conv'
        - 'TransformerConv' (тоже поддерживает edge_dim)
    """
    def __init__(self, hidden_dim, dropout, conv_type, heads, edge_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.conv_type = conv_type

        # Выбор слоя с правильными параметрами
        if conv_type == 'GATConv':
            self.msg = GATConv(
                hidden_dim, hidden_dim, heads=heads, concat=False,
                edge_dim=edge_dim, add_self_loops=False
            )
        elif conv_type == 'GATv2Conv':
            self.msg = GATv2Conv(
                hidden_dim, hidden_dim, heads=heads, concat=False,
                edge_dim=edge_dim, add_self_loops=False
            )
        elif conv_type == 'TransformerConv':
            # TransformerConv не имеет параметра add_self_loops
            self.msg = TransformerConv(
                hidden_dim, hidden_dim, heads=heads, concat=False,
                edge_dim=edge_dim
            )
        else:
            raise ValueError(f'Unsupported conv_type: {conv_type}')

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # FFN (размер расширения можно сделать параметром, пока зафиксируем 4)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        # Message passing с pre‑norm и residual
        x_norm = self.norm1(x)
        msg_out = self.msg(x_norm, edge_index, edge_attr)
        x = x + self.dropout(msg_out)

        # FFN с pre‑norm и residual
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        return x


# ----------------------------------------------------------------------
# Гибкая гетерогенная модель
# ----------------------------------------------------------------------
class GNNPlusHeteroFlex(nn.Module):
    """
    Модель с гибкими параметрами для использования в Optuna.
    """
    def __init__(self, cell_features, well_features, hidden_dim, out_seq_len=24,
                 num_phases=3, edge_dim=1, num_layers=3, dropout=0.1,
                 conv_type='GATConv', heads=1):
        super().__init__()
        self.cell_emb = Linear(cell_features, hidden_dim)
        self.well_emb = Linear(well_features, hidden_dim)
        self.edge_emb = Linear(edge_dim, hidden_dim)

        # Слои для графа ячеек
        self.cell_layers = nn.ModuleList([
            GNNPlusLayerFlex(hidden_dim, dropout, conv_type, heads, edge_dim=hidden_dim)
            for _ in range(num_layers)
        ])

        # Слой для связи ячейка → скважина (без edge_attr)
        if conv_type == 'GATConv':
            well_layer = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, add_self_loops=False)
        elif conv_type == 'GATv2Conv':
            well_layer = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, add_self_loops=False)
        elif conv_type == 'TransformerConv':
            well_layer = TransformerConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        else:
            raise ValueError(f'Unsupported conv_type: {conv_type}')

        self.well_conv = HeteroConv({
            ('cell', 'linked_to', 'well'): well_layer
        }, aggr='mean')

        # Финальный MLP
        self.well_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_phases * out_seq_len)
        )

    def forward(self, data):
        x_dict = {
            'cell': self.cell_emb(data['cell'].x),
            'well': self.well_emb(data['well'].x)
        }
        c2c_edge_index = data['cell', 'flows_to', 'cell'].edge_index
        c2c_edge_attr = data['cell', 'flows_to', 'cell'].edge_attr
        if c2c_edge_attr is not None:
            c2c_edge_attr = self.edge_emb(c2c_edge_attr)

        h_cell = x_dict['cell']
        for layer in self.cell_layers:
            h_cell = layer(h_cell, c2c_edge_index, c2c_edge_attr)
        x_dict['cell'] = h_cell

        well_updates = self.well_conv(x_dict, data.edge_index_dict)
        h_well = well_updates['well']
        out = self.well_mlp(h_well)
        return out.view(-1, 3, 24)  # out_seq_len = 24


# ----------------------------------------------------------------------
# Функция для обучения одной конфигурации (вызывается Optuna)
# ----------------------------------------------------------------------
def train_one_config(config, train_loader, val_loader, device, trial=None, verbose=True):
    """
    Обучает модель с заданными гиперпараметрами.
    Возвращает минимальную валидационную потерю (MSE) за эпохи.
    """
    # Параметры из config
    conv_type = config['conv_type']
    heads = config['heads']
    hidden_dim = config['hidden_dim']
    num_layers = config['num_layers']
    dropout = config['dropout']
    lr = config['lr']
    weight_decay = config['weight_decay']
    epochs = config.get('epochs', 12)
    patience = config.get('patience', 4)

    # Определяем размерности (берём из первого батча)
    sample_data, _ = next(iter(train_loader))
    cell_feat = sample_data['cell'].x.size(1)
    well_feat = sample_data['well'].x.size(1)
    edge_dim = 1   # у вас всегда 1 (TRAN)

    if verbose:
        print(f"\n[Training] Trial {trial.number if trial else 0}: {conv_type}, heads={heads}, hidden={hidden_dim}, layers={num_layers}, dropout={dropout:.3f}, lr={lr:.2e}, wd={weight_decay:.2e}")

    model = GNNPlusHeteroFlex(
        cell_features=cell_feat,
        well_features=well_feat,
        hidden_dim=hidden_dim,
        out_seq_len=24,
        num_phases=3,
        edge_dim=edge_dim,
        num_layers=num_layers,
        dropout=dropout,
        conv_type=conv_type,
        heads=heads
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # Train
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

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for v_batch, _ in val_loader:
                v_batch = v_batch.to(device)
                v_pred = model(v_batch)
                v_loss = criterion(v_pred, v_batch['well'].y)
                total_val_loss += v_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        # Вывод лога, если verbose
        if verbose:
            print(f"  Epoch {epoch:02d}/{epochs}: train MSE = {avg_train_loss:.6f}, val MSE = {avg_val_loss:.6f}")

        # Ранняя остановка и логирование для Optuna
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    Early stopping after {epoch} epochs (no improvement for {patience} epochs)")
                break

        if trial is not None:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                if verbose:
                    print("    Pruned by Optuna.")
                raise optuna.TrialPruned()

    if verbose:
        print(f"  Best val loss: {best_val_loss:.6f}\n")
    return best_val_loss


# ----------------------------------------------------------------------
# Функция цели (objective) для Optuna
# ----------------------------------------------------------------------
def objective(trial, train_loader, val_loader, device):
    # Определяем пространство поиска
    conv_type = trial.suggest_categorical('conv_type', ['GATConv', 'GATv2Conv', 'TransformerConv'])
    heads = trial.suggest_int('heads', 1, 4)
    hidden_dim = trial.suggest_int('hidden_dim', 64, 128, step=64)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    config = {
        'conv_type': conv_type,
        'heads': heads,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'lr': lr,
        'weight_decay': weight_decay,
        'epochs': 12,      # максимальное число эпох для поиска
        'patience': 4
    }

    val_loss = train_one_config(config, train_loader, val_loader, device, trial, verbose=True)
    return val_loss


# ----------------------------------------------------------------------
# Основная функция
# ----------------------------------------------------------------------
def main():
    # Загружаем конфигурацию проекта (пути и т.д.)
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Работаем на: {device}")

    # Загружаем данные (один раз)
    train_loader, val_loader, feat_list = load_graph_data(
        config['paths'],
        config['preprocess'],
        config['train']
    )
    print(f"Train графов: {len(train_loader.dataset)}, Val графов: {len(val_loader.dataset)}")

    # Создаём Optuna study
    study = optuna.create_study(
        direction='minimize',
        study_name='gnn_hyperopt',
        storage=None,          # можно указать SQLite для сохранения прогресса
        load_if_exists=False
    )

    # Запускаем поиск
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, device),
        n_trials=50,          # количество проб
        timeout=None
    )

    # Результаты
    print("\n=== Результаты поиска ===")
    print(f"Число завершённых попыток: {len(study.trials)}")
    best_trial = study.best_trial
    print(f"Лучшее значение (val MSE): {best_trial.value:.6f}")
    print("Лучшие гиперпараметры:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Сохраняем лучшие параметры в файл
    best_params = best_trial.params
    with open("best_params.yaml", "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)
    print("\nЛучшие параметры сохранены в best_params.yaml")


if __name__ == "__main__":
    main()