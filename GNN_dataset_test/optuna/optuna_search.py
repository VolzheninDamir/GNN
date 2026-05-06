import optuna
from optuna.trial import TrialState
import torch
import yaml
import os
import numpy as np
from torch_geometric.loader import DataLoader
from gnn_plus_hetero_optuna import GNNPlusHetero
from utils import load_graph_data

def objective(trial):
    # Загружаем базовую конфигурацию
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    # --- Пространство поиска гиперпараметров ---
    # Основные
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 2, 6)
    dropout = trial.suggest_float('dropout', 0.0, 0.3, step=0.1)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # Архитектурные – включаем все варианты
    conv_type = trial.suggest_categorical('conv_type',
                                          ['GAT', 'GATv2', 'GCNEdge', 'SAGEEdge',
                                           'GCN', 'SAGE', 'GIN'])

    gat_heads = 1
    use_edge_features = False  # значение по умолчанию

    if conv_type in ['GAT', 'GATv2']:
        gat_heads = trial.suggest_int('gat_heads', 1, 8, step=1)
        use_edge_features = trial.suggest_categorical('use_edge_features', [True, False])
    elif conv_type in ['GCNEdge', 'SAGEEdge']:
        use_edge_features = True  # эти слои всегда используют edge_attr
    # для остальных (GCN, SAGE, GIN) use_edge_features остаётся False

    well_aggr = trial.suggest_categorical('well_aggr', ['mean', 'sum', 'max'])
    ffn_expansion = trial.suggest_categorical('ffn_expansion', [2, 4])

    # Обновляем конфиг (часть параметров будет передана в модель напрямую)
    config['model']['nz'] = hidden_dim
    config['model']['num_layers'] = num_layers
    config['model']['dropout'] = dropout
    config['train']['warmup_learning_rate'] = lr
    config['train']['weight_decay'] = weight_decay

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTrial {trial.number} running on {device}")

    # Загрузка данных (можно закешировать, если не меняется batch_size)
    train_loader, val_loader, _ = load_graph_data(
        config['paths'],
        config['preprocess'],
        config['train']
    )

    # Получаем размерности
    sample_data, _ = next(iter(train_loader))
    cell_features = sample_data['cell'].x.size(1)
    well_features = sample_data['well'].x.size(1)
    edge_dim = 1
    if ('cell', 'flows_to', 'cell') in sample_data.edge_types:
        edge_attr = sample_data['cell', 'flows_to', 'cell'].get('edge_attr', None)
        if edge_attr is not None:
            if isinstance(edge_attr, torch.Tensor):
                edge_dim = edge_attr.size(1) if edge_attr.dim() == 2 else 1

    # Создаём модель с выбранными параметрами
    model = GNNPlusHetero(
        cell_features=cell_features,
        well_features=well_features,
        hidden_dim=hidden_dim,
        out_seq_len=24,
        num_phases=3,
        edge_dim=edge_dim,
        num_layers=num_layers,
        dropout=dropout,
        conv_type=conv_type,
        gat_heads=gat_heads,
        well_aggr=well_aggr,
        ffn_expansion=ffn_expansion,
        use_edge_features=use_edge_features
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    # Параметры обучения
    n_epochs = 10
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
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

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for v_batch, _ in val_loader:
                v_batch = v_batch.to(device)
                v_pred = model(v_batch)
                v_loss = criterion(v_pred, v_batch['well'].y)
                total_val_loss += v_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        # Сообщаем Optuna о текущем значении для pruning
        trial.report(avg_val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if epoch % 1 == 0:
            print(f"Epoch {epoch:02d}: Train MSE={avg_train_loss:.6f}, Val MSE={avg_val_loss:.6f}")

    return best_val_loss

if __name__ == "__main__":
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    study.optimize(objective, n_trials=30)

    print("\n=== Best trial ===")
    trial = study.best_trial
    print(f"Value (best val MSE): {trial.value:.6f}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # Сохраняем лучшие параметры
    with open("optuna_best_params.yaml", "w") as f:
        yaml.dump(trial.params, f)