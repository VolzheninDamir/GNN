import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv, GINConv, HeteroConv, Linear, MessagePassing
from torch_geometric.utils import degree

# ========== Обёртки для стандартных слоёв, чтобы они принимали edge_attr (игнорируя его) ==========
class GCNConvWithEdgeAttr(GCNConv):
    def forward(self, x, edge_index, edge_attr=None):
        return super().forward(x, edge_index)

class SAGEConvWithEdgeAttr(SAGEConv):
    def forward(self, x, edge_index, edge_attr=None):
        return super().forward(x, edge_index)

class GINConvWithEdgeAttr(GINConv):
    def forward(self, x, edge_index, edge_attr=None):
        return super().forward(x, edge_index)

# ========== Кастомные слои, использующие edge_attr ==========
class GCNEdgeConv(MessagePassing):
    """
    GCN с интеграцией признаков рёбер.
    """
    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.edge_lin = nn.Linear(edge_dim, out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, edge_attr):
        # edge_attr: [E, edge_dim]
        edge_attr = self.edge_lin(edge_attr)  # [E, out_channels]
        x = self.lin(x)                      # [N, out_channels]

        # Нормализация по степеням (как в GCN)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # [E]

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)
        out = out + self.bias
        return out

    def message(self, x_j, edge_attr, norm):
        # x_j: [E, out_channels], edge_attr: [E, out_channels]
        return (x_j + edge_attr) * norm.view(-1, 1)

class SAGEEdgeConv(MessagePassing):
    """
    GraphSAGE с интеграцией признаков рёбер.
    """
    def __init__(self, in_channels, out_channels, edge_dim, aggr='mean'):
        super().__init__(aggr=aggr)
        self.lin = nn.Linear(in_channels, out_channels)
        self.edge_lin = nn.Linear(edge_dim, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.lin(x)
        edge_attr = self.edge_lin(edge_attr)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

# ========== Фабрика свёрточных слоёв ==========
def build_conv(conv_type, hidden_dim, heads=1, use_edge_attr=True, edge_dim=1):
    if conv_type == 'GAT':
        return GATConv(hidden_dim, hidden_dim, heads=heads, concat=False,
                       edge_dim=hidden_dim if use_edge_attr else None,
                       add_self_loops=False)
    elif conv_type == 'GATv2':
        return GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False,
                         edge_dim=hidden_dim if use_edge_attr else None,
                         add_self_loops=False)
    elif conv_type == 'GCNEdge':
        return GCNEdgeConv(hidden_dim, hidden_dim, edge_dim=edge_dim)
    elif conv_type == 'SAGEEdge':
        return SAGEEdgeConv(hidden_dim, hidden_dim, edge_dim=edge_dim, aggr='mean')
    elif conv_type == 'GCN':
        return GCNConvWithEdgeAttr(hidden_dim, hidden_dim)
    elif conv_type == 'SAGE':
        return SAGEConvWithEdgeAttr(hidden_dim, hidden_dim)
    elif conv_type == 'GIN':
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        return GINConvWithEdgeAttr(mlp, train_eps=False)
    else:
        raise ValueError(f"Unknown conv_type: {conv_type}")

# ========== Базовый блок GNN+ ==========
class GNNPlusLayer(nn.Module):
    """
    Один блок GNN+ с возможностью использования разных свёрток.
    """
    def __init__(self, hidden_dim, dropout, conv_module, use_edge_attr=True, ffn_expansion=4):
        super().__init__()
        self.conv = conv_module
        self.use_edge_attr = use_edge_attr
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_expansion),
            nn.ReLU(),
            nn.Linear(hidden_dim * ffn_expansion, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        x_norm = self.norm1(x)
        # Все свёртки теперь принимают edge_attr (обёртки игнорируют его)
        msg_out = self.conv(x_norm, edge_index, edge_attr)
        x = x + self.dropout(msg_out)

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        return x

# ========== Основная гетерогенная модель ==========
class GNNPlusHetero(nn.Module):
    def __init__(self,
                 cell_features,
                 well_features,
                 hidden_dim=128,
                 out_seq_len=24,
                 num_phases=3,
                 edge_dim=1,
                 num_layers=3,
                 dropout=0.1,
                 conv_type='GAT',
                 gat_heads=1,
                 well_aggr='mean',
                 ffn_expansion=4,
                 use_edge_features=True):
        super().__init__()

        self.use_edge_features = use_edge_features
        self.is_custom_edge_layer = conv_type in ['GCNEdge', 'SAGEEdge']

        self.cell_emb = Linear(cell_features, hidden_dim)
        self.well_emb = Linear(well_features, hidden_dim)
        if use_edge_features and not self.is_custom_edge_layer:
            # Для кастомных слоёв edge_emb не нужен, они сами проецируют
            self.edge_emb = Linear(edge_dim, hidden_dim)
        else:
            self.edge_emb = None

        # Создаём список свёрточных слоёв
        convs = []
        for _ in range(num_layers):
            conv = build_conv(conv_type, hidden_dim, heads=gat_heads,
                              use_edge_attr=use_edge_features,
                              edge_dim=edge_dim)
            convs.append(conv)

        # Оборачиваем каждый слой в GNNPlusLayer
        self.cell_layers = nn.ModuleList([
            GNNPlusLayer(hidden_dim, dropout, convs[i],
                         use_edge_attr=use_edge_features,
                         ffn_expansion=ffn_expansion)
            for i in range(num_layers)
        ])

        # Слой для связи ячейка → скважина (без атрибутов рёбер)
        self.well_conv = HeteroConv({
            ('cell', 'linked_to', 'well'): GATConv(hidden_dim, hidden_dim,
                                                    heads=1, concat=False,
                                                    add_self_loops=False)
        }, aggr=well_aggr)

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

        if self.use_edge_features and c2c_edge_attr is not None:
            if self.is_custom_edge_layer:
                # Кастомные слои сами обрабатывают edge_attr (передаём исходные)
                pass
            else:
                # Для GAT/GATv2 проецируем edge_attr в hidden_dim
                c2c_edge_attr = self.edge_emb(c2c_edge_attr)
        else:
            c2c_edge_attr = None

        h_cell = x_dict['cell']
        for layer in self.cell_layers:
            h_cell = layer(h_cell, c2c_edge_index, c2c_edge_attr)
        x_dict['cell'] = h_cell

        well_updates = self.well_conv(x_dict, data.edge_index_dict)
        h_well = well_updates['well']

        out = self.well_mlp(h_well)
        return out.view(-1, 3, 24)  # out_seq_len = 24