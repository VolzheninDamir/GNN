
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

from utils import New_GeoData

import torch
from torch_geometric.data import HeteroData

from resdata.summary import Summary

import warnings
warnings.filterwarnings("ignore")

def prepare_grid_data(grdecl_folder, grid_names):

    grid_data = {}

    for grid_name in grid_names:
        grid_data[grid_name] = New_GeoData({'dataset_folder': grdecl_folder, 'grid_filename': grid_name})

    return grid_data


def load_one_case(static_feature_names, dynamic_feature_names, load_full_grid, use_labels, final_dir, rewrite_path, graph, multiply_features = False):

    summary = Summary.load(os.path.join(final_dir, "result.SMSPEC"), os.path.join(final_dir, "result.UNSMRY"))

    data = HeteroData()
    
    graph.update_graph_from_folder(
                                os.path.join(final_dir, "props"),
                                static_feature_names,
                                os.path.join(final_dir, "result.UNRST"),
                                dynamic_feature_names
                                   )

    if load_full_grid:
        graph.active = np.ones(shape=graph.active.shape)

    node_features_list = set(static_feature_names).union(dynamic_feature_names) - {'NTG'}

    edge_index, edge_index_local, edge_features, node_features, node_ladels = graph.get_graf_edges_fast(slices=(), return_tran=True, node_features_list=node_features_list, return_labels=use_labels)

    x = torch.tensor(np.expand_dims(np.prod(node_features, axis=1), 1) if multiply_features else node_features , dtype=torch.float32)                 # (N_nodes, 5)
    data['cell'].x = x

    edge_index = torch.tensor(edge_index.transpose(), dtype=torch.long)  # (2,E)
    data['cell', 'flows_to', 'cell'].edge_index = edge_index.T

    data['cell', 'flows_to', 'cell'].edge_attr = edge_features

    src_wells = []
    dst_cells = []
    wells_prod = []

    for well_ind, well in enumerate([x for x in summary.wells() if x.startswith('WELL')]):
        
        well_idxs = []
        for key in summary.keys():
            if key.startswith(f'COPT:{well}') and ',' not in key.split(':')[2]:
                well_idxs.append(int(key.split(':')[2])-1)
                
        dst_cells += well_idxs
        src_wells += [well_ind] * len(well_idxs)

        wells_prod.append(np.zeros((3 ,len(summary.get_days()))))
        for fluid_idx, kw in enumerate(['WOPT', 'WWPT', 'WGPT']):
            wells_prod[well_ind][fluid_idx] += summary.get_values(f'{kw}:{well}')

    data['well'].x = torch.zeros(len(wells_prod), 1)  # без признаков (или можно задать эмбеддинги)

    edge_index_well_cell = torch.tensor([dst_cells, src_wells], dtype=torch.long)
    data['cell', 'linked_to', 'well'].edge_index = edge_index_well_cell

    data['well'].y = torch.tensor(np.array(wells_prod), dtype=torch.float32)


    if use_labels:
        data['cell'].labels = torch.tensor(node_ladels, dtype=torch.float32) # (N_nodes, 1)

    model = os.path.basename(os.path.normpath(final_dir))

    print(model, 'Done!')

    torch.save(data, os.path.join(rewrite_path, f"{model}.pt"))

    return {'MODEL':[model],'MIN':[np.nanmin(x[node_ladels == 1], axis=0)],'MAX':[np.nanmax(x[node_ladels == 1], axis=0)],'MEAN':[np.nanmean(x[node_ladels == 1], axis=0)],'STD':[np.nanstd(x[node_ladels == 1], axis=0)],}

def run(config_path: str):

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    configs_preproc = cfg["preprocess"]
    configs_paths = cfg["paths"]
    
    os.makedirs(configs_paths['processed_data'], exist_ok=True)
    os.makedirs(os.path.join(configs_paths['processed_data'], 'samples'), exist_ok=True)
    os.makedirs(os.path.join(configs_paths['processed_data'], 'metadata'), exist_ok=True)

    metadata = pd.read_csv(os.path.join(configs_paths['raw_data'], 'metadata', 'metadata.csv'), sep = ';')

    grid_data = prepare_grid_data(os.path.join(configs_paths['raw_data'], 'samples', 'grdecl'), metadata['GRID'].unique())
    
    tasks = []
    procesesed_metadata = pd.DataFrame(columns=['MODEL','STATUS','MIN','MAX'])

    with ThreadPoolExecutor(max_workers=configs_preproc['max_workers']) as exe:

        for _, row in metadata.iterrows():
            fut = exe.submit(load_one_case,
                            configs_preproc['static_features'],
                            configs_preproc['dynamic_features'],
                            configs_preproc['load_full_grid'],
                            configs_preproc['use_labels'],
                            os.path.join(configs_paths['raw_data'], 'samples', row['MODEL']),
                            os.path.join(configs_paths['processed_data'], 'samples'),
                            grid_data[row['GRID']].copy(),
                            configs_preproc['multiply_features']
                            )
            tasks.append(fut)

        for meta_dict in as_completed(tasks):
            try:
                # print("Готово:", meta_dict.result()['MODEL'][0])
                tmp = meta_dict.result()
                tmp['STATUS'] = ['COMPLETE']
            except Exception as e:
                # print("Ошибка:", e)
                tmp = meta_dict.result()
                tmp['STATUS'] = ['ERROR']
            procesesed_metadata = pd.concat([procesesed_metadata, pd.DataFrame(tmp)])

            
    procesesed_metadata.to_csv(os.path.join(configs_paths['processed_data'], 'metadata', 'metadata.csv'), index=False)

if __name__ == "__main__":
    run('params.yaml')
    # import sys
    # run(sys.argv[1])