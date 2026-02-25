import xtgeo
import pyvista as pv
import os

# Пути (подставьте свои)
grdecl_file = 'C:/Users/Damir/Desktop/GNN_dataset_test/GNN_dataset_test/raw/samples/grdecl/e1_v00001.grdecl'
props_folder = 'C:/Users/Damir/Desktop/GNN_dataset_test/GNN_dataset_test/raw/samples/grdecl'

# Загружаем сетку
grid = xtgeo.grid_from_file(grdecl_file, fformat='grdecl')

# Загружаем свойство NTG, передавая имя и сетку
ntg_file = os.path.join(props_folder, 'e1_v00001.grdecl')
ntg_prop = xtgeo.gridproperty_from_file(
    ntg_file, 
    fformat='grdecl', 
    name='NTG',          # обязательно указываем имя свойства
    grid=grid            # можно передать сетку для проверки размерности
)

# Добавляем свойство в сетку для визуализации
grid.props['NTG'] = ntg_prop

# Визуализация
p = pv.Plotter()
grid.plot(plotter=p, property='NTG', cmap='jet', show_edges=True, opacity=0.6)
p.show()