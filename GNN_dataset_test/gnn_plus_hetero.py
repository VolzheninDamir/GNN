import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, Linear

class GNNPlusLayer(nn.Module):
    """
    Один блок архитектуры GNN+:
    - Message Passing (GATConv с признаками рёбер)
    - Pre‑norm, Dropout, Residual
    - FFN (MLP) с Pre‑norm и Residual
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.msg = GATConv(hidden_dim, hidden_dim, heads=1, concat=False,
                           edge_dim=hidden_dim, add_self_loops=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        # 1. Message Passing + Residual (Pre‑Norm)
        x_norm = self.norm1(x)
        msg_out = self.msg(x_norm, edge_index, edge_attr)
        x = x + self.dropout(msg_out)

        # 2. FFN + Residual (Pre‑Norm)
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        return x


class GNNPlusHetero(nn.Module):
    """
    Гетерогенная GNN+ для предсказания дебитов скважин.
    - Линейные проекции признаков узлов и рёбер
    - Несколько слоёв GNNPlusLayer для обработки ячеек (рёбра cell→cell)
    - Один слой для агрегации от ячеек к скважинам (cell→well)
    - Финальный MLP для скважин
    """
    def __init__(self,
                 cell_features,
                 well_features,
                 hidden_dim=128,
                 out_seq_len=25,
                 num_phases=3,
                 edge_dim=1,
                 num_layers=3,
                 dropout=0.1):
        super().__init__()

        # 1. Эмбеддинги узлов и рёбер
        self.cell_emb = Linear(cell_features, hidden_dim)
        self.well_emb = Linear(well_features, hidden_dim)
        self.edge_emb = Linear(edge_dim, hidden_dim)

        # 2. Слои для графа ячеек (cell → cell)
        self.cell_layers = nn.ModuleList([
            GNNPlusLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])

        # 3. Слой для связи ячейка → скважина (без атрибутов рёбер)
        self.well_conv = HeteroConv({
            ('cell', 'linked_to', 'well'): GATConv(hidden_dim, hidden_dim,
                                                    heads=1, concat=False,
                                                    add_self_loops=False)
        }, aggr='mean')

        # 4. Финальный MLP для скважин
        self.well_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_phases * out_seq_len)
        )

    def forward(self, data):
        # Проецируем признаки
        x_dict = {
            'cell': self.cell_emb(data['cell'].x),
            'well': self.well_emb(data['well'].x)
        }

        # Признаки рёбер между ячейками
        c2c_edge_index = data['cell', 'flows_to', 'cell'].edge_index
        c2c_edge_attr = data['cell', 'flows_to', 'cell'].edge_attr
        if c2c_edge_attr is not None:
            c2c_edge_attr = self.edge_emb(c2c_edge_attr)

        # Последовательно применяем слои GNN+ только к ячейкам
        h_cell = x_dict['cell']
        for layer in self.cell_layers:
            h_cell = layer(h_cell, c2c_edge_index, c2c_edge_attr)
        x_dict['cell'] = h_cell

        # Агрегация от ячеек к скважинам
        well_updates = self.well_conv(x_dict, data.edge_index_dict)
        h_well = well_updates['well']

        # Финальный прогноз
        out = self.well_mlp(h_well)
        return out.view(-1, 3, 25)