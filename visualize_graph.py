import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

# Загружаем данные
data = torch.load('C:/Users/Damir/Desktop/GNN_dataset_test/GNN_dataset_test/processed/samples/e1_v00004.pt', weights_only=False)

# Создаём подграф только для ячеек (игнорируем скважины)
# Для этого нужно извлечь рёбра между ячейками и соответствующие узлы
edge_index = data['cell', 'flows_to', 'cell'].edge_index

# Преобразуем в граф NetworkX (без параметра to_undirected, так как он уже неориентированный? 
# edge_index хранит каждое ребро в обе стороны? Проверим, если да, то можно просто передать)
# В PyG edge_index обычно хранит каждое ребро один раз (направленное), но для неориентированного графа 
# нужно либо добавить обратные рёбра, либо использовать to_undirected=True. Но раз мы строим только ячейки,
# можно использовать to_networkx с параметром to_undirected=True, но тогда нужно передать только данные ячеек.
# Проще создать граф вручную:
G = nx.Graph()
num_cells = data['cell'].num_nodes
G.add_nodes_from(range(num_cells))
# Добавляем рёбра (каждое ребро из edge_index)
edges = edge_index.t().tolist()
G.add_edges_from(edges)

# Если граф слишком большой, возьмём случайные 200 узлов
#if G.number_of_nodes() > 200:
    #import random
    #nodes_sample = random.sample(list(G.nodes), 200)
    #G = G.subgraph(nodes_sample)

plt.figure(figsize=(10, 8))
nx.draw(G, with_labels=False, node_size=20, node_color='blue', edge_color='gray')
plt.title("Граф ячеек (случайные 200 узлов)")
plt.show()