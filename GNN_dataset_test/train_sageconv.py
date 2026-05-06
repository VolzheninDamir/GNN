# train_sageconv.py
import torch
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Linear, HeteroConv, GATv2Conv
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn

from utils import load_graph_data

# ----------------------------------------------------------------------
# Кастомный SAGEConv с поддержкой edge features
# ----------------------------------------------------------------------
class SAGEConvWithEdge(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=None, aggr='mean', bias=True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim

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

    def forward(self, x, edge_index, edge_attr=None):
        x = self.lin(x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
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
# Блок GNN+ с SAGEConvWithEdge
# ----------------------------------------------------------------------
class GNNPlusLayerSAGE(nn.Module):
    def __init__(self, hidden_dim, dropout, edge_dim):
        super().__init__()
        self.msg = SAGEConvWithEdge(hidden_dim, hidden_dim, edge_dim=edge_dim, aggr='mean')
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
# Гетерогенная модель с SAGEConvWithEdge для ячеек и GATv2Conv для связи cell->well
# ----------------------------------------------------------------------
class GNNPlusHeteroSAGE(nn.Module):
    def __init__(self, cell_features, well_features, hidden_dim, out_seq_len=24,
                 num_phases=3, edge_dim=1, num_layers=3, dropout=0.2, heads=3):
        super().__init__()
        self.cell_emb = Linear(cell_features, hidden_dim)
        self.well_emb = Linear(well_features, hidden_dim)
        self.edge_emb = Linear(edge_dim, hidden_dim)

        self.cell_layers = nn.ModuleList([
            GNNPlusLayerSAGE(hidden_dim, dropout, edge_dim=hidden_dim)
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
# Обучение
# ----------------------------------------------------------------------
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

    # Получаем образец
    sample_data, _ = next(iter(train_loader))

    # Размерность edge_attr (из edge_feature_list)
    edge_dim = len(config['preprocess']['edge_feature_list'])  # теперь ['TRAN', 'DIST'] -> 2
    print(f"Размерность edge_attr: {edge_dim}")

    # Инициализация модели с параметрами из yaml
    model = GNNPlusHeteroSAGE(
        cell_features=sample_data['cell'].x.size(1),
        well_features=sample_data['well'].x.size(1),
        hidden_dim=config['model']['nz'],
        out_seq_len=24,
        num_phases=3,
        edge_dim=edge_dim,
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        heads=config['model'].get('heads', 3)   # добавим heads в yaml, если нет – 3
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['train']['warmup_learning_rate'],
        weight_decay=config['train']['weight_decay']
    )
    criterion = torch.nn.MSELoss()

    epochs = config['train']['epochs']
    patience = config['train'].get('patience', 4)
    train_history = []
    val_history = []

    print(f"Начинаем обучение на {len(train_loader.dataset)} графах...")

    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # Тренировка
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

        # Валидация
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for v_batch, _ in val_loader:
                v_batch = v_batch.to(device)
                v_pred = model(v_batch)
                v_loss = criterion(v_pred, v_batch['well'].y)
                total_val_loss += v_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_history.append(avg_val_loss)

        print(f"Эпоха {epoch:03d} | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")

        # Ранняя остановка
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            # Сохраняем лучшую модель
            torch.save(model.state_dict(), os.path.join(config['paths']['models'], 'best_sageconv_model.pth'))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Ранняя остановка на эпохе {epoch}")
                break

    # Сохраняем финальную модель (можно и лучшую, но для отчета сохраним финальную)
    os.makedirs(config['paths']['models'], exist_ok=True)
    save_path = os.path.join(config['paths']['models'], 'sageconv_oil_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Модель сохранена в {save_path}")

    # График
    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label='Train Loss')
    plt.plot(val_history, label='Val Loss')
    plt.yscale('log')
    plt.xlabel('Эпоха')
    plt.ylabel('MSE (нормализованный)')
    plt.legend()
    plt.title('Процесс обучения SAGEConv')
    plt.show()


if __name__ == "__main__":
    train_model()