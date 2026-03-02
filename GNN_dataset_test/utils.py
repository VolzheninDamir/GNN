import os
import gc
import glob
import random 
import numpy as np
import pandas as pd

import copy

import xtgeo
import pyvista as pv
import resfo

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch.amp import autocast
from torch.autograd import Variable
from torch.distributions import Normal
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

def ConvertTokens_fast(line):
    """
    Expands tokens of the type N*data to N copies of data.
    Возвращает list[str].
    """
    out = []
    for tok in line.split():
        if '*' in tok:
            n, val = tok.split('*', 1)
            out.extend([val] * int(n))
        else:
            out.append(tok)
    return out

def read_3d_file(path):
    """
    Opens file, searches for keyword section, parses into NumPy array.
    """
    buf = []
    with open(path) as gridfile:
        for line in gridfile:
            if line.startswith('--') or not line.strip() or line.startswith('NOECHO'):
                continue
            vals = ConvertTokens_fast(line)
            buf.extend(vals)
            if buf and buf[-1] == '/':
                buf.pop(0)
                buf.pop(-1)
                return np.array(buf, dtype=np.float64)
        return np.array([], dtype=np.float64)

# Хелпер: проверить совпадение граней пачкой
def faces_equal(face_pts_A, face_pts_B, eps):
    # face_pts_*: (..., 4, 3)
    diff = face_pts_A - face_pts_B
    d = np.linalg.norm(diff, axis=-1)   # (..., 4)
    return np.all(d < eps, axis=-1)     # (...,)

# Хелпер: проверить пересечение граней пачкой
def faces_equal_z(face_pts_A, face_pts_B):
    return np.any((face_pts_A[:, ::3, 2] < np.repeat(face_pts_B[:,::3, 2].max(axis = 1), 2).reshape(-1, 2)) & (face_pts_A[:, ::3, 2] > np.repeat(face_pts_B[:,::3, 2].min(axis = 1), 2).reshape(-1, 2)) | (face_pts_A[:, 1:3, 2] < np.repeat(face_pts_B[:,1:3, 2].max(axis = 1), 2).reshape(-1, 2)) & (face_pts_A[:, 1:3, 2] > np.repeat(face_pts_B[:,1:3, 2].min(axis = 1), 2).reshape(-1, 2)), axis=1)

class New_GeoData:
    def __init__(self, structure_dataset):

        self.points_edges = [(0,1,3,2), (4,5,7,6), (0,2,6,4), (1,3,7,5), (0,1,5,4), (2,3,7,6)]
        self.edges_nbh = {'down':(0,1), 'up':(1,0), 'left':(2,3), 'right':(3,2), 'front':(4,5), 'back':(5,4)}
        self.shifts = {'down' : (0, 0, -1),'up' : (0, 0, 1),'left' : (-1, 0, 0),'right' : (1, 0, 0),'front' : (0, -1, 0),'back' : (0, 1, 0),}
        self.directions = {'down' : 2,'up' : 2,'left' : 0,'right' : 0,'front' : 1,'back' : 1,}

        self.eps_div = 1e-8

        self.dataset_folder = structure_dataset['dataset_folder']
        self.grid_filename = structure_dataset['grid_filename']

        self.pts_coord = self.get_pts_coord()

        self.cell_ijk = np.stack(np.meshgrid(np.arange(self.k_max),np.arange(self.j_max),np.arange(self.i_max),indexing="ij"), axis=-1).reshape(-1, 3)[:, ::-1]

        self.active = None
    
    def copy(self):
        """Удобный метод для shallow copy"""
        return copy.copy(self)
    
    def __copy__(self):
        """Shallow copy"""
        cls = self.__class__
        result = cls.__new__(cls)
        result.points_edges = self.points_edges
        result.edges_nbh = self.edges_nbh
        result.shifts = self.shifts
        result.directions = self.directions
        result.eps_div = self.eps_div
        result.dataset_folder = self.dataset_folder
        result.grid_filename = self.grid_filename
        result.pts_coord = self.pts_coord
        result.cell_ijk = self.cell_ijk
        result.active = self.active
        result.i_max = self.i_max
        result.j_max = self.j_max
        result.k_max = self.k_max
        result.grid_idx = self.grid_idx

        return result
    
    def __deepcopy__(self, memo):
        """Deep copy: рекурсивно копирует вложенные объекты"""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.points_edges = copy.deepcopy(self.points_edges, memo)
        result.edges_nbh = copy.deepcopy(self.edges_nbh, memo)
        result.shifts = copy.deepcopy(self.shifts, memo)
        result.directions = copy.deepcopy(self.directions, memo)
        result.eps_div = copy.deepcopy(self.eps_div, memo)
        result.dataset_folder = copy.deepcopy(self.dataset_folder, memo)
        result.grid_filename = copy.deepcopy(self.grid_filename, memo)
        result.pts_coord = copy.deepcopy(self.pts_coord, memo)
        result.cell_ijk = copy.deepcopy(self.cell_ijk, memo)
        result.active = copy.deepcopy(self.active, memo)
        result.i_max = copy.deepcopy(self.i_max, memo)
        result.j_max = copy.deepcopy(self.j_max, memo)
        result.k_max = copy.deepcopy(self.k_max, memo)
        result.grid_idx = copy.deepcopy(self.grid_idx, memo)

        return result
    
    def get_pts_coord(self):
        
        mygrid = xtgeo.grid_from_file(os.path.join(self.dataset_folder, self.grid_filename), fformat='grdecl')

        dim, crn, inactind = mygrid.get_vtk_geometries() # Тут Вытаскивается размер сетки, количество вершин и активные ячейки 

        i_max, j_max, k_max = dim
        self.i_max = int(i_max-1)
        self.j_max = int(j_max-1)
        self.k_max = int(k_max-1)
        self.grid_idx = np.arange(self.i_max*self.j_max*self.k_max).reshape(self.k_max, self.j_max, self.i_max).transpose(-1,1,0)

        # Добавляем данные о пористости
        grd = pv.ExplicitStructuredGrid(dim, crn)
        grd.hide_cells(inactind, inplace=True)

        # Пересчитываем связность и визуализируем
        grd = grd.compute_connectivity()

        VTK_TO_TNAV = [0, 1, 3, 2, 4, 5, 7, 6]

        pts = np.array(grd.points.reshape(self.k_max, self.j_max, self.i_max, 8, 3)).transpose(2, 1, 0, 3, 4)[:, :, :, VTK_TO_TNAV, :]
        return pts
    
    def update_graph_from_folder(self,
                                 folder_with_properties,
                                 static_properties_names,
                                 unrst_path,
                                 dynamic_properties_names,
                                 ):

        self.props = dict()
        for prop_file_name in os.listdir(folder_with_properties):
            prop_keyword = prop_file_name.split('_')[-1].split('.')[0]
            if prop_keyword in static_properties_names:
                self.props[prop_keyword] = read_3d_file(os.path.join(folder_with_properties, prop_file_name)).reshape(self.k_max, self.j_max, self.i_max).transpose(-1,1,0)

        default_rquired_props = set(['NTG'])

        if len(set(self.props.keys()) & default_rquired_props) != len(default_rquired_props):
            raise ValueError(f'Вам не хватает {default_rquired_props - set(self.props.keys()) & default_rquired_props} для построения графа.')
        
        self.active = self.props['NTG'].copy()

        act = self.active.ravel(order="F")

        tmp_props = [[] for _ in dynamic_properties_names]
        props_count = np.zeros(len(dynamic_properties_names))

        for kw, arr in resfo.read(unrst_path):

            for ind_kw, n_kw in enumerate(dynamic_properties_names): # сохраняем нужные массивы по ключевым словам.

                if n_kw in kw: # выбор нужгного ключевого слова из всех ключевых слов

                    if props_count[ind_kw] == 0: # проверка, что ранее не брали это ключевое слово (вытаскиваем первый timestep)

                        arr1 = list(arr.copy())

                        for i in act: # сохраняем по активным ячейкам 
                            if i == 0:
                                tmp_props[ind_kw].append(np.nan)
                            if i == 1:
                                tmp_props[ind_kw].append(arr1.pop(0))
                        tmp_props[ind_kw] = np.array(tmp_props[ind_kw])
                        props_count[ind_kw] += 1

        for count_ind, count in enumerate(props_count):
            if count == 0:
                if dynamic_properties_names[count_ind] == 'SGAS':

                    gas_calc_rquired_props = set(['NTG'])
                    if len(set(self.props.keys()) & gas_calc_rquired_props) != len(gas_calc_rquired_props):
                        raise ValueError(f'Вам не хватает {gas_calc_rquired_props - set(self.props.keys()) & gas_calc_rquired_props} для расчета SGAS.')
        
                    print(f"SGAS определен как 1 - SWAT - SOIL.")
                    self.props[dynamic_properties_names[count_ind]] = (1 - (tmp_props[dynamic_properties_names.index('SWAT')] + tmp_props[dynamic_properties_names.index('SOIL')]).round(3)).reshape(self.k_max, self.j_max, self.i_max).transpose(-1,1,0)
                else:
                    raise ValueError(f"{dynamic_properties_names[count_ind]} не найдено в файле .UNRST.")
            else:
                self.props[dynamic_properties_names[count_ind]] = tmp_props[count_ind].reshape(self.k_max, self.j_max, self.i_max).transpose(-1,1,0)

    def find_cells_intersection(self, mode, # ['regular', 'fault']
                                I0, J0, K0,
                                I1, J1, K1,
                                faces,
                                side,
                                ):
        edges_src = []
        edges_dst = []
        edges_area = []
        edges_direction = []
        m = self.active_slice_mask[I0, J0, K0] & self.active_slice_mask[I1, J1, K1]

        m_ = self.active_slice_mask[I0, J0, K0]
        A = self.pts_coord[I0[m_], J0[m_], K0[m_]][:, faces[self.edges_nbh[side][0], :]]
        B = self.pts_coord[I1[m_], J1[m_], K1[m_]][:, faces[self.edges_nbh[side][1], :]]
        ok_ = faces_equal(A, B, self.eps_div)

        if np.any(m): # проверка что есть хотябы одна пара активных ячеек
            A = self.pts_coord[I0[m], J0[m], K0[m]][:, faces[self.edges_nbh[side][0], :]]  # сторона 1
            B = self.pts_coord[I1[m], J1[m], K1[m]][:, faces[self.edges_nbh[side][1], :]]  # сторона 2
            if mode == 'regular':
                ok = faces_equal(A, B, self.eps_div)
            elif mode == 'fault':
                ok = faces_equal_z(A, B)

            if np.any(ok):
                edges_src.append(self.grid_idx[I0[m][ok], J0[m][ok], K0[m][ok]])
                edges_dst.append(self.grid_idx[I1[m][ok], J1[m][ok], K1[m][ok]])

                face_pts = self.pts_coord[I0[m], J0[m], K0[m]][:, faces[self.edges_nbh[side][0],:]]
                edges_direction.extend([self.directions[side]]*len(face_pts))
                face_center = face_pts.mean(axis=1)
                v1 = face_pts[:,1] - face_pts[:,0]
                v2 = face_pts[:,3] - face_pts[:,0]
                A_vec = np.cross(v1, v2)

                cell_center_i = self.pts_coord[I0[m], J0[m], K0[m]].mean(axis=1)
                cell_center_j = self.pts_coord[I1[m], J1[m], K1[m]].mean(axis=1)

                D_i = face_center - cell_center_i
                D_j = face_center - cell_center_j

                A_dot_Di = np.einsum('ij,ij->i', A_vec, D_i)
                Di_dot_Di = np.einsum('ij,ij->i', D_i, D_i)
                A_dot_Dj = np.einsum('ij,ij->i', A_vec, D_j)
                Dj_dot_Dj = np.einsum('ij,ij->i', D_j, D_j)

                # сохранить для последующего вычисления TRANX
                edges_area.append((A_dot_Di, Di_dot_Di, A_dot_Dj, Dj_dot_Dj))


        return edges_src, edges_dst, edges_area, edges_direction, ok_, m_
    
    def get_graf_edges_fast(self, slices = (), return_tran = False, node_features_list = None, return_labels=False):

        """
        Быстрое построение графа по полному совпадению граней с 6 ближайшими индексными соседями.
        Возвращает edge_index формы (2, E) в глобальных индексах.
        """
        default_rquired_props = set(['NTG'])

        if len(set(self.props.keys()) & default_rquired_props) != len(default_rquired_props):
            raise ValueError(f'Вам не хватает {default_rquired_props - set(self.props.keys()) & default_rquired_props} для построения графа.')
        
        I, J, K = self.i_max, self.j_max, self.k_max

        # Номера точек граней
        faces = np.array(self.points_edges, dtype=int)

        edges_src = []
        edges_dst = []
        edges_area = []
        edges_direction = []

        self.active_slice_mask = np.zeros(self.active.shape).astype(bool)
        self.active_slice_mask[slices] = self.active[slices].astype(bool)


        for side in ['right','back']: # пробегаем по направлениям

            di, dj, dk = self.shifts[side]

            i = np.arange(0, I - di)
            j = np.arange(0, J - dj)
            k = np.arange(0, K)
            I0, J0, K0 = np.meshgrid(i, j, k, indexing="ij")
            I1, J1, K1 = I0 + di, J0 + dj, K0
            
            edges_src_, edges_dst_, edges_area_, edges_direction_, ok_, m_ = self.find_cells_intersection('regular',  I0, J0, K0, I1, J1, K1, faces, side, )
            edges_src.extend(edges_src_)
            edges_dst.extend(edges_dst_)
            edges_area.extend(edges_area_)
            edges_direction.extend(edges_direction_)

            temp = self.grid_idx[I0[m_][~ok_], J0[m_][~ok_], K0[m_][~ok_]]

            for i in temp:
                i_, j_, k_ = self.cell_ijk[i]

                if i_+2 > self.i_max:
                    continue

                I0, J0, K0 = np.meshgrid([i_], [j_], [k_], indexing="ij")
                I0 = np.repeat(I0, K, axis=-1)
                J0 = np.repeat(J0, K, axis=-1)
                K0 = np.repeat(K0, K, axis=-1)

                i = np.arange(i_+di, i_+1+di)
                j = np.arange(j_+dj, j_+1+dj)
                k = np.arange(0, K)
                I1, J1, K1 = np.meshgrid(i, j, k, indexing="ij")

                edges_src_, edges_dst_, edges_area_, edges_direction_, _, _ = self.find_cells_intersection('fault',  I0, J0, K0, I1, J1, K1, faces, side, )
                edges_src.extend(edges_src_)
                edges_dst.extend(edges_dst_)
                edges_area.extend(edges_area_)
                edges_direction.extend(edges_direction_)

        # --- собрать все рёбра ---
        if len(edges_src) == 0:
            return np.empty((2, 0), dtype=int)

        src = np.concatenate(edges_src)
        dst = np.concatenate(edges_dst)

        areas = np.concatenate([np.array(edges_area[iii]).T for iii in range(len(edges_area))])
        direct = np.array(edges_direction)

        # делаем (min,max), чтобы убрать дубликаты
        ab = np.stack([np.minimum(src, dst), np.maximum(src, dst)], axis=1)

        # уникальные рёбра + индексы откуда они взялись
        ab_unique, unique_idx = np.unique(ab, axis=0, return_index=True)

        edge_index = ab_unique.T        # (2, E)
        edges_area_unique = areas[unique_idx]  # площади к уникальным рёбрам
        edges_direction = direct[unique_idx]

        edges_area_unique = abs(edges_area_unique.reshape(edges_area_unique.shape[0], 2, 2))

        # глобальные вершины, встречающиеся в edge_index
        unique_nodes = np.unique(edge_index)   
        mapping_arr = np.full(edge_index.max()+1, -1, dtype=int)
        mapping_arr[unique_nodes] = np.arange(len(unique_nodes))
        edge_index_local = mapping_arr[edge_index]
        ijk = self.cell_ijk[unique_nodes].T

        ### экспорт свойств грида
        node_features = None
        if node_features_list is not None:
            if len(set(self.props.keys()) & set(node_features_list)) != len(node_features_list):
                raise ValueError(f'В графе нет свойств {set(node_features_list) - set(self.props.keys()) & set(node_features_list)} которые вы запрашиваете.')


            node_features = np.stack([
                self.props[prop_name][tuple(ijk)]
             for prop_name in node_features_list], axis=1)

        ### свойства грида

        ### метки коллектора
        node_labels = None
        if return_labels:
            node_labels = self.props['NTG'][tuple(ijk)]
        ### метки коллектора

        ### проводимость
        edge_features = None
        if return_tran:

            rquired_props_for_tran_calk = set(['PERMX', 'PERMY', 'PERMZ'])

            if len(set(self.props.keys()) & rquired_props_for_tran_calk) != len(rquired_props_for_tran_calk):
                raise ValueError(f'Вам не хватает {rquired_props_for_tran_calk - set(self.props.keys()) & rquired_props_for_tran_calk} для расчета проводимости.')

            ijk1 = np.array([self.cell_ijk[p] for p in edge_index[0]])  # (E, 3)
            ijk2 = np.array([self.cell_ijk[p] for p in edge_index[1]])  # (E, 3)

            perm1 = np.choose(edges_direction, [self.props['PERMX'][tuple(ijk1.T)], self.props['PERMY'][tuple(ijk1.T)], self.props['PERMZ'][tuple(ijk1.T)]])  # (E,)
            perm2 = np.choose(edges_direction, [self.props['PERMX'][tuple(ijk2.T)], self.props['PERMY'][tuple(ijk2.T)], self.props['PERMZ'][tuple(ijk2.T)]])  # (E,)

            A1 = edges_area_unique[:,0,0]
            D1 = edges_area_unique[:,0,1]
            A2 = edges_area_unique[:,1,0]
            D2 = edges_area_unique[:,1,1]

            data_for_tran_calc = np.stack([
                perm1 * self.props['NTG'][tuple(ijk1.T)] * A1/D1,
                perm2 * self.props['NTG'][tuple(ijk2.T)] * A2/D2
            ], axis=1).T   # (E, 2, 3)

            CDARCY = 0.00852702

            edge_features = CDARCY/(1/data_for_tran_calc[0] + 1/data_for_tran_calc[1]) #TRAN
        ### проводимость

        return edge_index, edge_index_local, edge_features, node_features, node_labels

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import glob

# Предполагается, что функция set_seed определена выше
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class III_stage_Dataset(Dataset):
    """
    Класс Dataset для загрузки графов из .pt файлов с возможностью нормализации:
    - признаков ячеек (cell.x)
    - целевой переменной скважин (well.y)
    - атрибутов рёбер (edge_attr) для всех типов рёбер
    """
    def __init__(self,
                 file_paths,
                 scaler='norm',  # для cell.x ['norm', 'stand', None]
                 global_min=None,
                 global_max=None,
                 global_mean=None,
                 global_std=None,
                 # Параметры для well.y
                 y_scaler=None,      # если None, используется значение scaler
                 y_global_min=None,
                 y_global_max=None,
                 y_global_mean=None,
                 y_global_std=None,
                 # Параметры для edge_attr
                 edge_scaler=None,
                 edge_global_min=None,
                 edge_global_max=None,
                 edge_global_mean=None,
                 edge_global_std=None,
                 ):
        # Фильтруем пустые файлы
        self.files = [f for f in file_paths if os.path.getsize(f) > 0]
        self.eps = 1e-8

        # Настройки для cell.x
        self.scaler = scaler
        self.global_min = global_min
        self.global_max = global_max
        self.global_mean = global_mean
        self.global_std = global_std

        # Настройки для well.y
        self.y_scaler = y_scaler if y_scaler is not None else scaler
        self.y_global_min = y_global_min
        self.y_global_max = y_global_max
        self.y_global_mean = y_global_mean
        self.y_global_std = y_global_std

        # Настройки для edge_attr
        self.edge_scaler = edge_scaler if edge_scaler is not None else scaler
        self.edge_global_min = edge_global_min
        self.edge_global_max = edge_global_max
        self.edge_global_mean = edge_global_mean
        self.edge_global_std = edge_global_std

        # Валидация для cell.x
        if self.scaler == 'norm':
            if self.global_min is not None and self.global_max is not None:
                self._validate_normalization()
            else:
                raise AssertionError('Для cell.x не установлены min/max')
        elif self.scaler == 'stand':
            if self.global_mean is not None and self.global_std is not None:
                self._validate_standardization()
            else:
                raise AssertionError('Для cell.x не установлены mean/std')

        # Валидация для well.y
        if self.y_scaler == 'norm':
            if self.y_global_min is not None and self.y_global_max is not None:
                self._validate_y_normalization()
            else:
                # Если нет статистик, просто не применяем нормализацию (или можно предупредить)
                pass
        elif self.y_scaler == 'stand':
            if self.y_global_mean is not None and self.y_global_std is not None:
                self._validate_y_standardization()
            else:
                pass

        # Валидация для edge_attr
        if self.edge_scaler == 'norm':
            if self.edge_global_min is not None and self.edge_global_max is not None:
                self._validate_edge_normalization()
        elif self.edge_scaler == 'stand':
            if self.edge_global_mean is not None and self.edge_global_std is not None:
                self._validate_edge_standardization()

    def _validate_normalization(self):
        """Проверка размерности статистик для cell.x (min/max)."""
        random_sample = self[random.randint(0, len(self.files)-1)][0]['cell']
        feat_dim = random_sample.x.shape[1]
        if self.global_min.shape[1] != feat_dim or self.global_min.shape[0] != 1:
            raise AssertionError(f'global_min имеет размерность {self.global_min.shape}, ожидается (1, {feat_dim})')
        if self.global_max.shape[1] != feat_dim or self.global_max.shape[0] != 1:
            raise AssertionError(f'global_max имеет размерность {self.global_max.shape}, ожидается (1, {feat_dim})')

    def _validate_standardization(self):
        """Проверка размерности статистик для cell.x (mean/std)."""
        random_sample = self[random.randint(0, len(self.files)-1)][0]['cell']
        feat_dim = random_sample.x.shape[1]
        if self.global_mean.shape[1] != feat_dim or self.global_mean.shape[0] != 1:
            raise AssertionError(f'global_mean имеет размерность {self.global_mean.shape}, ожидается (1, {feat_dim})')
        if self.global_std.shape[1] != feat_dim or self.global_std.shape[0] != 1:
            raise AssertionError(f'global_std имеет размерность {self.global_std.shape}, ожидается (1, {feat_dim})')

    def _validate_y_normalization(self):
        """Проверка размерности статистик для well.y (min/max). Ожидается (1, 3, 1) или (1, 3)."""
        random_sample = self[random.randint(0, len(self.files)-1)][0]
        if 'well' not in random_sample.node_types or not hasattr(random_sample['well'], 'y'):
            return
        # Статистики должны транслироваться: можно хранить (1, 3, 1)
        if self.y_global_min.dim() == 2:
            # (1, 3) -> расширяем до (1, 3, 1)
            self.y_global_min = self.y_global_min.unsqueeze(-1)
            self.y_global_max = self.y_global_max.unsqueeze(-1)
        if self.y_global_min.shape != (1, 3, 1):
            raise AssertionError(f'y_global_min имеет размерность {self.y_global_min.shape}, ожидается (1, 3, 1)')

    def _validate_y_standardization(self):
        """Проверка размерности статистик для well.y (mean/std). Ожидается (1, 3, 1)."""
        random_sample = self[random.randint(0, len(self.files)-1)][0]
        if 'well' not in random_sample.node_types or not hasattr(random_sample['well'], 'y'):
            return
        if self.y_global_mean.dim() == 2:
            self.y_global_mean = self.y_global_mean.unsqueeze(-1)
            self.y_global_std = self.y_global_std.unsqueeze(-1)
        if self.y_global_mean.shape != (1, 3, 1):
            raise AssertionError(f'y_global_mean имеет размерность {self.y_global_mean.shape}, ожидается (1, 3, 1)')

    def _validate_edge_normalization(self):
        """Проверка размерности статистик для edge_attr (min/max). Ожидается (1, feat_dim)."""
        random_sample = self[random.randint(0, len(self.files)-1)][0]
        # Найдём первый тип рёбер с атрибутами
        feat_dim = None
        for edge_type in random_sample.edge_types:
            if hasattr(random_sample[edge_type], 'edge_attr'):
                edge_attr = random_sample[edge_type].edge_attr
                if isinstance(edge_attr, np.ndarray):
                    feat_dim = 1 if edge_attr.ndim == 1 else edge_attr.shape[1]
                else:
                    feat_dim = 1 if edge_attr.dim() == 1 else edge_attr.size(1)
                break
        if feat_dim is None:
            return  # нет атрибутов рёбер
        if self.edge_global_min is not None:
            if self.edge_global_min.shape != (1, feat_dim):
                raise AssertionError(f'edge_global_min имеет размерность {self.edge_global_min.shape}, ожидается (1, {feat_dim})')
        if self.edge_global_max is not None:
            if self.edge_global_max.shape != (1, feat_dim):
                raise AssertionError(f'edge_global_max имеет размерность {self.edge_global_max.shape}, ожидается (1, {feat_dim})')

    def _validate_edge_standardization(self):
        """Проверка размерности статистик для edge_attr (mean/std). Ожидается (1, feat_dim)."""
        random_sample = self[random.randint(0, len(self.files)-1)][0]
        feat_dim = None
        for edge_type in random_sample.edge_types:
            if hasattr(random_sample[edge_type], 'edge_attr'):
                edge_attr = random_sample[edge_type].edge_attr
                if isinstance(edge_attr, np.ndarray):
                    feat_dim = 1 if edge_attr.ndim == 1 else edge_attr.shape[1]
                else:
                    feat_dim = 1 if edge_attr.dim() == 1 else edge_attr.size(1)
                break
        if feat_dim is None:
            return
        if self.edge_global_mean is not None:
            if self.edge_global_mean.shape != (1, feat_dim):
                raise AssertionError(f'edge_global_mean имеет размерность {self.edge_global_mean.shape}, ожидается (1, {feat_dim})')
        if self.edge_global_std is not None:
            if self.edge_global_std.shape != (1, feat_dim):
                raise AssertionError(f'edge_global_std имеет размерность {self.edge_global_std.shape}, ожидается (1, {feat_dim})')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        graph = torch.load(self.files[idx], weights_only=False)

        # Замена NaN на 0 для признаков ячеек
        if 'cell' in graph.node_types and hasattr(graph['cell'], 'x'):
            graph['cell'].x = torch.nan_to_num(graph['cell'].x, nan=0.0)

        # Замена NaN на 0 для целевой переменной скважин (если вдруг есть)
        if 'well' in graph.node_types and hasattr(graph['well'], 'y'):
            graph['well'].y = torch.nan_to_num(graph['well'].y, nan=0.0)

        # Нормализация cell.x
        if self.scaler == 'norm':
            if self.global_min is not None and self.global_max is not None:
                graph['cell'].x = ((graph['cell'].x - self.global_min) /
                                   (self.global_max - self.global_min + self.eps))
        elif self.scaler == 'stand':
            if self.global_mean is not None and self.global_std is not None:
                graph['cell'].x = ((graph['cell'].x - self.global_mean) /
                                   (self.global_std + self.eps))

        # Нормализация well.y (если существует)
        if 'well' in graph.node_types and hasattr(graph['well'], 'y'):
            if self.y_scaler == 'norm':
                if self.y_global_min is not None and self.y_global_max is not None:
                    graph['well'].y = ((graph['well'].y - self.y_global_min) /
                                       (self.y_global_max - self.y_global_min + self.eps))
            elif self.y_scaler == 'stand':
                if self.y_global_mean is not None and self.y_global_std is not None:
                    graph['well'].y = ((graph['well'].y - self.y_global_mean) /
                                       (self.y_global_std + self.eps))

        if torch.isnan(graph['well'].y).any():
            print(f"ВНИМАНИЕ: well.y содержит NaN после нормализации в файле {self.files[idx]}")

        # Нормализация атрибутов рёбер (для всех типов рёбер)
        for edge_type in graph.edge_types:
            edge_attr = graph[edge_type].get('edge_attr', None)
            if edge_attr is not None:
                # Преобразуем numpy в тензор, если необходимо
                if isinstance(edge_attr, np.ndarray):
                    edge_attr = torch.from_numpy(edge_attr).float()
                    graph[edge_type].edge_attr = edge_attr
                # Замена NaN на 0
                edge_attr = torch.nan_to_num(edge_attr, nan=0.0)
                # Нормализация
                if self.edge_scaler == 'norm':
                    if self.edge_global_min is not None and self.edge_global_max is not None:
                        graph[edge_type].edge_attr = ((edge_attr - self.edge_global_min) /
                                                       (self.edge_global_max - self.edge_global_min + self.eps))
                elif self.edge_scaler == 'stand':
                    if self.edge_global_mean is not None and self.edge_global_std is not None:
                        graph[edge_type].edge_attr = ((edge_attr - self.edge_global_mean) /
                                                       (self.edge_global_std + self.eps))

        sample_name = os.path.basename(os.path.normpath(self.files[idx]))
        return graph, sample_name


def compute_y_stats(file_paths, scaler_type):
    """
    Вычисляет глобальные статистики для well.y по всем файлам.
    Возвращает кортеж (min, max, mean, std) в виде тензоров (1, 3, 1) или None.
    """
    all_y = []
    for path in file_paths:
        data = torch.load(path, weights_only=False)
        if 'well' in data.node_types and hasattr(data['well'], 'y'):
            all_y.append(data['well'].y)  # каждый имеет форму (n_wells, 3, 25)
    if not all_y:
        return None, None, None, None

    # Объединяем все скважины по первому измерению
    all_y = torch.cat(all_y, dim=0)  # (total_wells, 3, 25)

    if scaler_type == 'norm':
        # Минимум и максимум по скважинам и времени для каждой фазы
        y_min = all_y.amin(dim=(0, 2), keepdim=True)  # (1, 3, 1)
        y_max = all_y.amax(dim=(0, 2), keepdim=True)
        return y_min, y_max, None, None
    elif scaler_type == 'stand':
        # Среднее и стандартное отклонение по скважинам и времени для каждой фазы
        y_mean = all_y.mean(dim=(0, 2), keepdim=True)   # (1, 3, 1)
        y_std = all_y.std(dim=(0, 2), keepdim=True) + 1e-8
        return None, None, y_mean, y_std
    else:
        return None, None, None, None


def compute_edge_stats(file_paths, scaler_type, edge_type=('cell', 'flows_to', 'cell')):
    """
    Вычисляет глобальные статистики для edge_attr указанного типа рёбер по всем файлам.
    Возвращает кортеж (min, max, mean, std) в виде тензоров (1, feat_dim) или None.
    По умолчанию ожидается, что edge_attr одномерный (feat_dim=1).
    """
    all_edge_attrs = []
    for path in file_paths:
        data = torch.load(path, weights_only=False)
        if edge_type in data.edge_types and hasattr(data[edge_type], 'edge_attr'):
            edge_attr = data[edge_type].edge_attr
            # Преобразуем numpy в тензор, если нужно
            if isinstance(edge_attr, np.ndarray):
                edge_attr = torch.from_numpy(edge_attr).float()
            all_edge_attrs.append(edge_attr)  # каждый имеет форму (E,) или (E, feat_dim)

    if not all_edge_attrs:
        return None, None, None, None

    # Объединяем все рёбра по первому измерению
    all_edge_attrs = torch.cat(all_edge_attrs, dim=0)  # (total_edges, feat_dim) или (total_edges,)

    # Если одномерный, добавим размерность признака для единообразия
    if all_edge_attrs.dim() == 1:
        all_edge_attrs = all_edge_attrs.view(-1, 1)
    feat_dim = all_edge_attrs.size(1)

    if scaler_type == 'norm':
        e_min = all_edge_attrs.amin(dim=0, keepdim=True)  # (1, feat_dim)
        e_max = all_edge_attrs.amax(dim=0, keepdim=True)
        return e_min, e_max, None, None
    elif scaler_type == 'stand':
        e_mean = all_edge_attrs.mean(dim=0, keepdim=True)   # (1, feat_dim)
        e_std = all_edge_attrs.std(dim=0, keepdim=True) + 1e-8
        return None, None, e_mean, e_std
    else:
        return None, None, None, None


def load_graph_data(configs_paths, configs_preproc, configs_train, return_val=True):
    """
    Загружает данные, разделяет на train/val, вычисляет статистики нормализации
    для cell.x (из metadata), для well.y и edge_attr (из самих файлов), создаёт DataLoader'ы.
    """
    set_seed(configs_train["seed"])

    all_samples_paths = glob.glob(f"{os.path.join(configs_paths['processed_data'], 'samples')}/*")
    train_samples_paths, val_samples_paths = train_test_split(
        all_samples_paths,
        test_size=1 - configs_train['train_size'],
        random_state=configs_train['seed']
    )

    val_samples = list(map(os.path.basename, val_samples_paths))

    # Чтение метаданных для cell.x
    meta = pd.read_csv(os.path.join(configs_paths['processed_data'], 'metadata', 'metadata.csv'))
    meta = meta[meta['STATUS'] == 'COMPLETE']
    mask = ~meta['MODEL'].isin(val_samples)

    print("Метаданные (train часть):")
    print(meta[mask])

    # Определяем список признаков (исключая NTG)
    data_features = set(configs_preproc['static_features']).union(configs_preproc['dynamic_features']) - {'NTG'}
    if len(data_features) == 0:
        raise AssertionError("Нет признаков для обучения модели. Укажите хотя бы один признак в static_features или dynamic_features (кроме NTG).")
    if configs_preproc['multiply_features']:
        data_features = ['multed_features']

    data_features_num = len(data_features)

    # Статистики для cell.x (из метаданных)
    mins = torch.empty((1, data_features_num))
    maxs = torch.empty((1, data_features_num))
    means = torch.empty((1, data_features_num))
    stds = torch.empty((1, data_features_num))

    mins[0, :] = torch.tensor(
        np.array([np.fromstring(s.strip('[]'), sep=' ') for s in meta[mask]['MIN']]).min(axis=0),
        dtype=torch.float32
    )
    maxs[0, :] = torch.tensor(
        np.array([np.fromstring(s.strip('[]'), sep=' ') for s in meta[mask]['MAX']]).max(axis=0),
        dtype=torch.float32
    )

    means_arr = np.array([np.fromstring(s.strip('[]'), sep=' ') for s in meta[mask]['MEAN']])
    means[0, :] = torch.tensor(means_arr.mean(axis=0), dtype=torch.float32)
    stds_arr = np.array([np.fromstring(s.strip('[]'), sep=' ') for s in meta[mask]['STD']])
    std = np.sqrt((stds_arr**2 + (means_arr - means.numpy())**2).sum(axis=0) / len(means_arr))
    stds[0, :] = torch.tensor(std, dtype=torch.float32)

    # Статистики для well.y (только на тренировочной выборке)
    y_min, y_max, y_mean, y_std = compute_y_stats(train_samples_paths, configs_preproc['scaler_type'])

    # Статистики для edge_attr (только на тренировочной выборке)
    e_min, e_max, e_mean, e_std = compute_edge_stats(train_samples_paths, configs_preproc['scaler_type'])

    # Создание датасетов с передачей всех статистик
    train_dataset = III_stage_Dataset(
        train_samples_paths,
        scaler=configs_preproc['scaler_type'],
        global_min=mins,
        global_max=maxs,
        global_mean=means,
        global_std=stds,
        y_scaler=configs_preproc['scaler_type'],
        y_global_min=y_min,
        y_global_max=y_max,
        y_global_mean=y_mean,
        y_global_std=y_std,
        edge_scaler=configs_preproc['scaler_type'],
        edge_global_min=e_min,
        edge_global_max=e_max,
        edge_global_mean=e_mean,
        edge_global_std=e_std
    )

    val_dataset = III_stage_Dataset(
        val_samples_paths,
        scaler=configs_preproc['scaler_type'],
        global_min=mins,
        global_max=maxs,
        global_mean=means,
        global_std=stds,
        y_scaler=configs_preproc['scaler_type'],
        y_global_min=y_min,
        y_global_max=y_max,
        y_global_mean=y_mean,
        y_global_std=y_std,
        edge_scaler=configs_preproc['scaler_type'],
        edge_global_min=e_min,
        edge_global_max=e_max,
        edge_global_mean=e_mean,
        edge_global_std=e_std
    )

    train_dataloader = DataLoader(train_dataset, batch_size=configs_train['batch_size'], shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=configs_train['batch_size'], shuffle=False)

    if return_val:
        return train_dataloader, val_dataloader, list(data_features)
    else:
        dataset = III_stage_Dataset(
            all_samples_paths,
            scaler=configs_preproc['scaler_type'],
            global_min=mins,
            global_max=maxs,
            global_mean=means,
            global_std=stds,
            y_scaler=configs_preproc['scaler_type'],
            y_global_min=y_min,
            y_global_max=y_max,
            y_global_mean=y_mean,
            y_global_std=y_std,
            edge_scaler=configs_preproc['scaler_type'],
            edge_global_min=e_min,
            edge_global_max=e_max,
            edge_global_mean=e_mean,
            edge_global_std=e_std
        )
        dataloader = DataLoader(dataset, batch_size=configs_train['batch_size'], shuffle=False)
        return dataloader, None, list(data_features)

"""
class III_stage_Dataset(Dataset):
    
    '''
    Этот класс представляет собой реализацию набора данных PyTorch для работы с .pt-файлами, 
    содержащими данные для обучения графового автоэнкодера
    
    '''
    def __init__(self,
                 file_paths, 
                 scaler = 'norm', # ['norm', 'stand'] 
                 global_min = None,
                 global_max = None,
                 global_mean = None,
                 global_std = None,
                 ):
        # Поиск и сортировка всех пакетных файлов в директории
        self.files = file_paths
        # Фильтрация пустых файлов
        self.files = [f for f in self.files if os.path.getsize(f) > 0]

        # Инициализация параметров нормализации
        self.scaler = scaler
        self.global_min = global_min
        self.global_max = global_max
        self.global_mean = global_mean
        self.global_std = global_std
        self.eps = 1e-8  # Малая константа для предотвращения деления на ноль

        if self.scaler == 'norm':
            if self.global_min is not None and self.global_max is not None:
                self.validate_normalization()
            else:
                raise AssertionError('min/max не установлены')

        elif self.scaler == 'stand':
            if self.global_mean is not None and self.global_std is not None:
                self.validate_standardtization()
            else:
                raise AssertionError('mean/std не установлены')

    def validate_normalization(self):
        random_sample = self[random.randint(0, len(self.files)-1)][0]['cell']
        if random_sample.x.shape[1] != self.global_min.shape[1] and self.global_min.shape[0] != 1:
            raise AssertionError(f'Неправильная размерность global_min {self.global_min.shape} (нужна (1, {random_sample.x.shape[1]})).')
        if random_sample.x.shape[1] != self.global_max.shape[1] and self.global_max.shape[0] != 1:
            raise AssertionError(f'Неправильная размерность global_max {self.global_max.shape} (нужна (1, {random_sample.x.shape[1]})).')
        
    def validate_standardtization(self):
        random_sample = self[random.randint(0, len(self.files)-1)][0]['cell']
        if random_sample.x.shape[1] != self.global_mean.shape[1] and self.global_mean.shape[0] != 1:
            raise AssertionError(f'Неправильная размерность global_mean {self.global_mean.shape} (нужна (1, {random_sample.x.shape[1]})).')
        if random_sample.x.shape[1] != self.global_std.shape[1] and self.global_std.shape[0] != 1:
            raise AssertionError(f'Неправильная размерность global_std {self.global_std.shape} (нужна (1, {random_sample.x.shape[1]})).')

    def __len__(self):
        return len(self.files)  # Возвращает количество пакетных файлов

    def __getitem__(self, idx):

        # Конвертация в тензор PyTorch
        graph = torch.load(self.files[idx], weights_only=False)  # [B, C, H, W]

        if self.scaler == 'norm':
            # Применение нормализации, если min/max установлены
            if self.global_min is not None and self.global_max is not None:
                graph['cell'].x = ((graph['cell'].x - self.global_min) / ( self.global_max - self.global_min + self.eps))
            else:
                raise AssertionError('min/max не установлены')
        elif self.scaler == 'stand':
            # Применение нормализации, если min/max установлены
            if self.global_mean is not None and self.global_std is not None:
                graph['cell'].x = ((graph['cell'].x - self.global_mean) / (self.global_std + self.eps))
            else:
                raise AssertionError('mean/std не установлены')

        sample_name = os.path.basename(os.path.normpath(self.files[idx]))

        return graph, sample_name  # Возвращает нормализованные данные и метаданные

def load_graph_data(configs_paths, configs_preproc, configs_train, return_val = True):

    set_seed(configs_train["seed"])

    all_samples_paths = glob.glob(f"{os.path.join(configs_paths['processed_data'], 'samples')}/*")
    train_samples_paths, val_samples_paths = train_test_split(all_samples_paths, test_size=1-configs_train['train_size'], random_state=configs_train['seed'])

    val_samples = list(map(os.path.basename, val_samples_paths))

    meta = pd.read_csv(os.path.join(configs_paths['processed_data'], 'metadata', 'metadata.csv'))
    meta = meta[meta['STATUS'] == 'COMPLETE'] 
    mask = ~meta['MODEL'].isin(val_samples)

    print(meta)

    data_features = set(configs_preproc['static_features']).union(configs_preproc['dynamic_features']) - {'NTG'}

    if len(data_features) == 0:
        raise AssertionError("Нет признаков для обучения модели, выберите хотя бы один признак из списка static_features или dynamic_features в params.yaml, кроме NTG")

    if configs_preproc['multiply_features']:
        data_features = ['multed_features']

    data_features_num = len(data_features)

    mins = torch.empty((1, data_features_num))
    maxs = torch.empty((1, data_features_num))
    means = torch.empty((1, data_features_num))
    stds = torch.empty((1, data_features_num))

    mins[0, :] = torch.tensor(np.array([np.fromstring(s.strip('[]'), sep=' ') for s in meta[mask]['MIN']]).min(axis=0), dtype=torch.float32)
    maxs[0, :] = torch.tensor(np.array([np.fromstring(s.strip('[]'), sep=' ') for s in meta[mask]['MAX']]).max(axis=0), dtype=torch.float32)

    means_arr = np.array([np.fromstring(s.strip('[]'), sep=' ') for s in meta[mask]['MEAN']])
    means[0, :] = torch.tensor(means_arr.mean(axis=0), dtype=torch.float32)
    stds_arr = np.array([np.fromstring(s.strip('[]'), sep=' ') for s in meta[mask]['STD']])
    std = np.sqrt((stds_arr**2 +(means_arr-means.numpy())**2).sum(axis = 0)/len(means_arr))
    del means_arr
    del stds_arr
    stds[0, :] = torch.tensor(std, dtype=torch.float32)

    if return_val:
        train_dataset = III_stage_Dataset(train_samples_paths, scaler = configs_preproc['scaler_type'], global_max=maxs, global_min = mins, global_mean=means, global_std = stds)
        val_dataset = III_stage_Dataset(val_samples_paths, scaler = configs_preproc['scaler_type'], global_max=maxs, global_min = mins, global_mean=means, global_std = stds)

        train_dataloader = DataLoader(train_dataset, batch_size=configs_train['batch_size'], shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=configs_train['batch_size'], shuffle=False)

        return train_dataloader, val_dataloader, data_features
    else:
        dataset = III_stage_Dataset(all_samples_paths, scaler = configs_preproc['scaler_type'], global_max=maxs, global_min = mins, global_mean=means, global_std = stds)
        dataloader = DataLoader(dataset, batch_size=configs_train['batch_size'], shuffle=False)
        return dataloader, None, data_features
"""