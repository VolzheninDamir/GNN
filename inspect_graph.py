import torch
from torch_geometric.utils import degree

file = r"C:/Users/Damir/Desktop/GNN_dataset_test/GNN_dataset_test/processed/samples/e1_v00001.pt"
data = torch.load(file, weights_only=False)

print(data)
print("Node types:", data.node_types)
print("Edge types:", data.edge_types)
print("Cell features shape:", data['cell'].x.shape)
print("Well target shape:", data['well'].y.shape)
print("Edges cell->cell:", data['cell', 'flows_to', 'cell'].edge_index.shape)
print("Edges cell->well:", data['cell', 'linked_to', 'well'].edge_index.shape)

# Посмотрим первые несколько значений признаков ячеек
print("First 5 cell features:\n", data['cell'].x[:5])