import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, Linear

class SimpleHeteroGNN(nn.Module):
    def __init__(self, cell_features, well_features, hidden_dim=64, out_seq_len=25, num_phases=3):
        super().__init__()
        # Входные линейные слои для каждого типа узлов
        self.cell_lin = Linear(cell_features, hidden_dim)
        self.well_lin = Linear(well_features, hidden_dim)

        # Слой для рёбер между ячейками (cell → cell)
        self.conv_cell = SAGEConv(hidden_dim, hidden_dim)

        # Слой для рёбер от ячеек к скважинам (cell → well)
        # SAGEConv с разными типами узлов принимает кортеж (src_x, dst_x)
        self.conv_well = SAGEConv((hidden_dim, hidden_dim), hidden_dim)

        # Выходной MLP для скважин
        self.well_mlp = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, num_phases * out_seq_len)
        )

    def forward(self, data):
        # Извлекаем данные из гетерогенного графа
        cell_x = data['cell'].x                     # (число_ячеек, cell_features)
        well_x = data['well'].x                     # (число_скважин, well_features)
        edge_index_cell = data['cell', 'flows_to', 'cell'].edge_index   # (2, E_cell)
        edge_index_well = data['cell', 'linked_to', 'well'].edge_index  # (2, E_well)

        # 1. Embedding
        cell_h = self.cell_lin(cell_x)               # (число_ячеек, hidden_dim)
        well_h = self.well_lin(well_x)               # (число_скважин, hidden_dim)

        # 2. Обновление ячеек через рёбра cell→cell
        cell_h = self.conv_cell(cell_h, edge_index_cell)

        # 3. Обновление скважин через рёбра cell→well
        # edge_index_well: первая строка – индексы ячеек, вторая – индексы скважин
        well_h = self.conv_well((cell_h, well_h), edge_index_well)

        # 4. Выходной MLP для скважин
        well_out = self.well_mlp(well_h)              # (число_скважин, 3*25)
        well_out = well_out.view(-1, 3, 25)           # (число_скважин, 3, 25)

        return well_out