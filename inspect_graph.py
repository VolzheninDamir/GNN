import torch
from torch_geometric.utils import degree

file = r"C:/Users/Damir/Desktop/GNN_dataset_test/GNN_dataset_test/processed/samples/e1_v00003.pt"
data = torch.load(file, weights_only=False)

# --- Проверка на NaN/Inf ---
print("Проверка на NaN/Inf:")
cell_nan = torch.isnan(data['cell'].x).any().item()
cell_inf = torch.isinf(data['cell'].x).any().item()
print(f"cell.x contains NaN: {cell_nan}")
print(f"cell.x contains Inf: {cell_inf}")

if cell_nan:
    nan_count_per_feature = torch.isnan(data['cell'].x).sum(dim=0)
    print("NaN count per feature:", nan_count_per_feature.tolist())

if 'well' in data.node_types and hasattr(data['well'], 'y'):
    well_nan = torch.isnan(data['well'].y).any().item()
    well_inf = torch.isinf(data['well'].y).any().item()
    print(f"well.y contains NaN: {well_nan}")
    print(f"well.y contains Inf: {well_inf}")
    if well_nan:
        well_nan_count = torch.isnan(data['well'].y).sum().item()
        print(f"Total NaNs in well.y: {well_nan_count}")
# ---------------------------------

print("\nОбщая информация о графе:")
print(data)
print("Node types:", data.node_types)
print("Edge types:", data.edge_types)
print("Cell features shape:", data['cell'].x.shape)
print("Well target shape:", data['well'].y.shape)
print("Edges cell->cell:", data['cell', 'flows_to', 'cell'].edge_index.shape)
print("Edges cell->well:", data['cell', 'linked_to', 'well'].edge_index.shape)

print("Edges attr cell->cell", data['cell', 'flows_to', 'cell'].edge_attrs)

# Первые несколько значений признаков ячеек
print("\nFirst 5 cell features:\n", data['cell'].x[:5])