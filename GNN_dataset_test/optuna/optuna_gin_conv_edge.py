# optuna_ginconv_edge_no_prune.py (исправленная версия с учётом реальной размерности рёбер)
import os
import torch
import torch.nn as nn
from torch_geometric.nn import Linear, HeteroConv, GATv2Conv
from torch_geometric.nn.conv import MessagePassing
import optuna
import yaml
from utils import load_graph_data

# ----------------------------------------------------------------------
# Кастомный GINConv с поддержкой edge features
# ----------------------------------------------------------------------
class GINConvWithEdge(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=None, eps=0.0, train_eps=False, bias=True, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.eps = torch.nn.Parameter(torch.Tensor([eps])) if train_eps else eps

        self.lin = Linear(in_channels, out_channels, bias=False)
        if edge_dim is not None:
            self.edge_lin = Linear(edge_dim, out_channels, bias=False)
        else:
            self.edge_lin = None

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.edge_lin is not None:
            self.edge_lin.reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        if isinstance(self.eps, torch.nn.Parameter):
            self.eps.data.fill_(0.0)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.lin(x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if isinstance(self.eps, torch.nn.Parameter):
            out = out + self.eps * x
        else:
            out = out + self.eps * x
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_j, edge_attr):
        if edge_attr is not None and self.edge_lin is not None:
            edge_msg = self.edge_lin(edge_attr)
            msg = x_j + edge_msg
        else:
            msg = x_j
        return msg


# ----------------------------------------------------------------------
# Блок GNN+ с GINConvWithEdge
# ----------------------------------------------------------------------
class GNNPlusLayerGIN(nn.Module):
    def __init__(self, hidden_dim, dropout, edge_dim):
        super().__init__()
        self.msg = GINConvWithEdge(hidden_dim, hidden_dim, edge_dim=edge_dim, eps=0.0, train_eps=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        x_norm = self.norm1(x)
        msg_out = self.msg(x_norm, edge_index, edge_attr)
        x = x + self.dropout(msg_out)
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        return x


# ----------------------------------------------------------------------
# Гетерогенная модель с GINConvWithEdge для ячеек и GATv2Conv для связи cell->well
# ----------------------------------------------------------------------
class GNNPlusHeteroGIN(nn.Module):
    def __init__(self, cell_features, well_features, hidden_dim, out_seq_len=24,
                 num_phases=3, edge_dim=1, num_layers=3, dropout=0.2, heads=3):
        super().__init__()
        self.cell_emb = Linear(cell_features, hidden_dim)
        self.well_emb = Linear(well_features, hidden_dim)
        self.edge_emb = Linear(edge_dim, hidden_dim)
        self.cell_layers = nn.ModuleList([
            GNNPlusLayerGIN(hidden_dim, dropout, edge_dim=hidden_dim)
            for _ in range(num_layers)
        ])
        well_layer = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, add_self_loops=False)
        self.well_conv = HeteroConv({
            ('cell', 'linked_to', 'well'): well_layer
        }, aggr='mean')
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
        return out.view(-1, 3, 24)


# ----------------------------------------------------------------------
# Функция обучения одной конфигурации (без Optuna pruning)
# ----------------------------------------------------------------------
def train_one_config(config, train_loader, val_loader, device, trial=None, verbose=True):
    hidden_dim = config['hidden_dim']
    num_layers = config['num_layers']
    dropout = config['dropout']
    lr = config['lr']
    weight_decay = config['weight_decay']
    heads = config['heads']
    edge_dim = config['edge_dim']          # <-- ИСПРАВЛЕНО: берём из config
    epochs = config.get('epochs', 20)
    patience = config.get('patience', 1)

    sample_data, _ = next(iter(train_loader))
    cell_feat = sample_data['cell'].x.size(1)
    well_feat = sample_data['well'].x.size(1)

    model = GNNPlusHeteroGIN(
        cell_features=cell_feat,
        well_features=well_feat,
        hidden_dim=hidden_dim,
        out_seq_len=24,
        num_phases=3,
        edge_dim=edge_dim,
        num_layers=num_layers,
        dropout=dropout,
        heads=heads
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    best_val_loss = float('inf')
    no_improve = 0

    if verbose:
        print(f"\n[Training] Trial {trial.number if trial else 0}: "
              f"hidden={hidden_dim}, layers={num_layers}, heads={heads}, "
              f"dropout={dropout:.3f}, lr={lr:.2e}, wd={weight_decay:.2e}, edge_dim={edge_dim}")

    for epoch in range(1, epochs + 1):
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

        if verbose:
            print(f"  Epoch {epoch:02d}/{epochs}: train MSE = {avg_train_loss:.6f}, val MSE = {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    Early stopping after {epoch} epochs (no improvement for {patience} epochs)")
                break

    if verbose:
        print(f"  Best val loss: {best_val_loss:.6f}\n")
    return best_val_loss


# ----------------------------------------------------------------------
# Objective для Optuna
# ----------------------------------------------------------------------
def objective(trial, train_loader, val_loader, device, edge_dim):
    hidden_dim = trial.suggest_int('hidden_dim', 64, 128, step=32)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.25, step=0.05)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    heads = trial.suggest_int('heads', 2, 4)

    config = {
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'lr': lr,
        'weight_decay': weight_decay,
        'heads': heads,
        'edge_dim': edge_dim,           # <-- ИСПРАВЛЕНО: передаём размерность рёбер
        'epochs': 20,
        'patience': 1
    }

    val_loss = train_one_config(config, train_loader, val_loader, device, trial=None, verbose=True)
    return val_loss


# ----------------------------------------------------------------------
# Основная функция
# ----------------------------------------------------------------------
def main():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Работаем на: {device}")

    train_loader, val_loader, feat_list = load_graph_data(
        config['paths'],
        config['preprocess'],
        config['train']
    )
    print(f"Train графов: {len(train_loader.dataset)}, Val графов: {len(val_loader.dataset)}")

    # Получаем реальную размерность признаков рёбер из конфига
    edge_dim = len(config['preprocess']['edge_feature_list'])   # <-- ИСПРАВЛЕНО
    print(f"Размерность edge_attr в поиске: {edge_dim}")

    study = optuna.create_study(
        direction='minimize',
        study_name='ginconv_edge_hyperopt_no_prune',
        pruner=optuna.pruners.NopPruner(),
        storage=None,
        load_if_exists=False
    )

    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, device, edge_dim),
        n_trials=30,
        timeout=None
    )

    print("\n=== Результаты поиска для GINConv с поддержкой рёбер ===")
    print(f"Число завершённых попыток: {len(study.trials)}")
    best_trial = study.best_trial
    print(f"Лучшее значение (val MSE): {best_trial.value:.6f}")
    print("Лучшие гиперпараметры:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    best_params = best_trial.params
    with open("best_params_ginconv_edge.yaml", "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)
    print("\nЛучшие параметры сохранены в best_params_ginconv_edge.yaml")


if __name__ == "__main__":
    main()