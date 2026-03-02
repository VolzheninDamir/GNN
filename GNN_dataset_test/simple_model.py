import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear

class SimpleHeteroGNN(nn.Module):
    def __init__(self, cell_features, well_features, hidden_dim=64, out_seq_len=25, num_phases=3):
        super().__init__()
        
        # 1. Эмбеддинги (приводим всё к hidden_dim)
        self.cell_emb = Linear(cell_features, hidden_dim)
        self.well_emb = Linear(well_features, hidden_dim)

        # 2. Несколько слоев для пласта (Cell -> Cell)
        # Увеличим глубину, чтобы информация "текла" дальше по кубу
        self.cell_convs = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim) for _ in range(3)
        ])

        # 3. Сбор данных на скважины (Cell -> Well)
        # Используем HeteroConv для передачи данных из ячеек в скважины
        self.well_conv = HeteroConv({
            ('cell', 'linked_to', 'well'): SAGEConv(hidden_dim, hidden_dim)
        }, aggr='mean')

        # 4. Прогнозный MLP
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
        
        # Связи
        c2c_edge_index = data['cell', 'flows_to', 'cell'].edge_index
        edge_index_dict = data.edge_index_dict

        # Шаг 1: Пропаганда внутри пласта (ячейка -> ячейка)
        h_cell = x_dict['cell']
        for conv in self.cell_convs:
            h_cell = conv(h_cell, c2c_edge_index)
            h_cell = F.relu(h_cell)
        
        x_dict['cell'] = h_cell

        # Шаг 2: Сбор информации на скважины (ячейка -> скважина)
        # well_updates получит эмбеддинги для узлов-приемников (скважин)
        well_updates = self.well_conv(x_dict, edge_index_dict)
        h_well = well_updates['well']

        # Шаг 3: Финальный слой
        out = self.well_mlp(h_well)
        return out.view(-1, 3, 25)