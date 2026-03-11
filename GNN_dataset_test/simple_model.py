import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, Linear

class SimpleHeteroGNN(nn.Module):
    def __init__(self, cell_features, well_features, hidden_dim=64, out_seq_len=25, num_phases=3, edge_dim=1):
        super().__init__()
        
        # 1. Эмбеддинги для узлов (приводим всё к hidden_dim)
        self.cell_emb = Linear(cell_features, hidden_dim)
        self.well_emb = Linear(well_features, hidden_dim)

        # 2. Эмбеддинг для атрибутов рёбер (проецируем исходные признаки в hidden_dim)
        self.edge_emb = Linear(edge_dim, hidden_dim)

        # 3. Несколько слоев для пласта (Cell -> Cell) с использованием атрибутов рёбер
        # Теперь edge_dim = hidden_dim, потому что мы применили эмбеддинг
        self.cell_convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=1, concat=False, edge_dim=hidden_dim, add_self_loops=False)
            for _ in range(3)
        ])

        # 4. Сбор данных на скважины (Cell -> Well) – здесь атрибуты рёбер не используются
        self.well_conv = HeteroConv({
            ('cell', 'linked_to', 'well'): GATConv(hidden_dim, hidden_dim, heads=1, concat=False, add_self_loops=False)
        }, aggr='mean')

        # 5. Прогнозный MLP
        self.well_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_phases * out_seq_len)
        )

    def forward(self, data):
        # Получаем начальные признаки
        x_dict = {
            'cell': self.cell_emb(data['cell'].x),
            'well': self.well_emb(data['well'].x)
        }
        
        # Связи и атрибуты рёбер для ячеек
        c2c_edge_index = data['cell', 'flows_to', 'cell'].edge_index
        c2c_edge_attr = data['cell', 'flows_to', 'cell'].get('edge_attr', None)
        if c2c_edge_attr is not None:
            # Применяем эмбеддинг к атрибутам рёбер
            c2c_edge_attr = self.edge_emb(c2c_edge_attr)
        edge_index_dict = data.edge_index_dict

        # Шаг 1: Пропаганда внутри пласта (ячейка -> ячейка)
        h_cell = x_dict['cell']
        for conv in self.cell_convs:
            if c2c_edge_attr is not None:
                h_cell = conv(h_cell, c2c_edge_index, edge_attr=c2c_edge_attr)
            else:
                h_cell = conv(h_cell, c2c_edge_index)
            h_cell = F.relu(h_cell)
        
        x_dict['cell'] = h_cell

        # Шаг 2: Сбор информации на скважины (ячейка -> скважина)
        well_updates = self.well_conv(x_dict, edge_index_dict)
        h_well = well_updates['well']

        # Шаг 3: Финальный слой
        out = self.well_mlp(h_well)
        return out.view(-1, 3, 25)