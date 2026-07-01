# GNN Dataset — прогноз дебитов скважин на графовых нейросетях

Проект для построения и обучения **гетерогенных графовых нейросетей (GNN)** на данных гидродинамического моделирования нефтяных пластов (ECLIPSE / GRDECL). Модель предсказывает **кумулятивные дебиты скважин** по трём фазам — нефть, вода, газ (`WOPT`, `WWPT`, `WGPT`) — на горизонте **24 временных шага**.

## Содержание

- [Идея проекта](#идея-проекта)
- [Структура репозитория](#структура-репозитория)
- [Формат данных](#формат-данных)
- [Графовое представление](#графовое-представление)
- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Конфигурация (`params.yaml`)](#конфигурация-paramsyaml)
- [Обучение моделей](#обучение-моделей)
- [Подбор гиперпараметров (Optuna)](#подбор-гиперпараметров-optuna)
- [Визуализация и отладка](#визуализация-и-отладка)
- [Зависимости](#зависимости)

---

## Идея проекта

Каждый вариант гидродинамического расчёта преобразуется в **гетерограф** PyTorch Geometric:

| Компонент | Описание |
|-----------|----------|
| **Узлы `cell`** | Ячейки 3D-сетки пласта с петрофизическими и динамическими свойствами |
| **Узлы `well`** | Скважины (без входных признаков, только связи с ячейками) |
| **Рёбра `cell → cell`** | Соседство ячеек с признаками `TRAN`, `DIST` |
| **Рёбра `cell → well`** | Связь скважины с ячейками перфорации |
| **Целевая переменная** | `well.y` — логарифмированные кумулятивные дебиты `(скважины × 3 фазы × 24 шага)` |

Архитектура **GNN+** (`gnn_plus_hetero_v2.py`) обрабатывает граф ячеек через несколько слоёв `TransformerConv`, агрегирует информацию на скважины и выдаёт временной ряд дебитов.

---

## Структура репозитория

```
GNN_dataset_test/
├── GNN_dataset_test/          # Основной код и данные
│   ├── raw/                   # Исходные данные симуляций
│   │   ├── metadata/
│   │   │   └── metadata.csv   # Список моделей и сеток
│   │   └── samples/
│   │       ├── e1_v00001/     # Результаты одного расчёта
│   │       │   ├── props/     # Статические свойства (.inc)
│   │       │   ├── result.SMSPEC
│   │       │   ├── result.UNRST
│   │       │   └── result.UNSMRY
│   │       └── grdecl/        # Файлы сетки (.grdecl)
│   ├── processed/             # Обработанные графы (.pt)
│   ├── models/                # Сохранённые веса моделей
│   ├── optuna/                # Скрипты и результаты HPO
│   ├── preprocess.py            # Препроцессинг raw → processed
│   ├── utils.py                 # Парсинг GRDECL, датасет, загрузка
│   ├── gnn_plus_hetero_v2.py    # Основная модель GNN+
│   ├── simple_model.py          # Базовая GAT-модель
│   ├── train_gnn_plus.py        # Обучение GNN+
│   ├── train.py                 # Обучение SimpleHeteroGNN
│   ├── train_sageconv.py        # Обучение SAGEConv-варианта
│   ├── params.yaml              # Конфигурация (не в git, см. ниже)
│   └── dataloaders.ipynb        # Исследовательский ноутбук
├── inspect_graph.py             # Проверка .pt-графов
├── visualize_graph.py           # Визуализация графа ячеек
├── visualize_3d.py              # 3D-визуализация сетки (xtgeo + pyvista)
└── README.md
```

---

## Формат данных

### `raw/metadata/metadata.csv`

Таблица со списком расчётов (разделитель `;`):

| Колонка | Описание |
|---------|----------|
| `MODEL` | Имя папки с результатами (например, `e1_v00001`) |
| `GRID` | Имя файла сетки в `raw/samples/grdecl/` |
| `Path`, `Status` | Служебные поля |

### Папка одного расчёта (`raw/samples/<MODEL>/`)

- **`props/`** — статические свойства ячеек в формате ECLIPSE `.inc` (`PORO`, `PERMX`, `NTG`, …)
- **`result.SMSPEC`**, **`result.UNSMRY`** — сводные данные по скважинам
- **`result.UNRST`** — динамические поля (`SWAT`, `SOIL`, `SGAS`)
- **`grdecl/`** — геометрия сетки (`.grdecl`)

### Результат препроцессинга (`processed/`)

- **`processed/samples/<MODEL>.pt`** — объект `HeteroData` (PyG)
- **`processed/metadata/metadata.csv`** — статистики нормализации (`MIN`, `MAX`, `MEAN`, `STD`) по каждому расчёту

> Данные (`raw/`, `processed/`, `*.pt`, `*.inc`, `*.grdecl`) исключены из git — их нужно подготовить локально.

---

## Графовое представление

```
┌─────────────────────────────────────────────────────────┐
│                    HeteroData                           │
├─────────────────────────────────────────────────────────┤
│  cell.x        — признаки ячеек (NTG, PORO, PERM*,      │
│                  SWAT, SOIL, SGAS, X, Y, Z, …)            │
│  cell.labels   — маска активных ячеек (NTG)             │
│  (cell, flows_to, cell)                                 │
│    .edge_index — связи между соседними ячейками         │
│    .edge_attr  — TRAN, DIST                             │
│  (cell, linked_to, well)                                │
│    .edge_index — ячейки перфорации → скважины           │
│  well.x        — заглушка (1 признак)                   │
│  well.y        — целевые дебиты (log1p, 3 × 24)         │
└─────────────────────────────────────────────────────────┘
```

Класс `New_GeoData` в `utils.py` отвечает за:

- чтение GRDECL-сетки и построение топологии ячеек;
- расчёт трансмиссивностей `TRAN` и расстояний `DIST` на рёбрах;
- загрузку статических и динамических свойств.

---

## Установка

### 1. Клонирование и виртуальное окружение

```bash
git clone <url-репозитория>
cd GNN_dataset_test

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Установка зависимостей

Рекомендуется сначала установить PyTorch с поддержкой CUDA (при наличии GPU), затем остальные пакеты:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install numpy pandas scikit-learn matplotlib networkx PyYAML
pip install optuna xtgeo pyvista resfo resdata
```

> Версии, проверенные в проекте: `torch==2.5.1+cu121`, `torch-geometric==2.7.0`, `resdata==6.2.5`, `xtgeo==4.18.0`.

### 3. Конфигурация

Файл `params.yaml` находится в `.gitignore`. Создайте его в папке `GNN_dataset_test/` по образцу из раздела [Конфигурация](#конфигурация-paramsyaml) и укажите **абсолютные пути** к вашим данным.

---

## Быстрый старт

Все команды выполняются из каталога `GNN_dataset_test/GNN_dataset_test/`:

```bash
cd GNN_dataset_test
```

### Шаг 1. Препроцессинг

```bash
python preprocess.py
```

Скрипт читает `params.yaml`, параллельно обрабатывает все модели из `metadata.csv` и сохраняет `.pt`-файлы в `processed/samples/`.

### Шаг 2. Обучение

```bash
python train_gnn_plus.py
```

Обучает модель `GNNPlusHetero` (v2) с взвешенной MSE-функцией потерь и early stopping. Веса сохраняются в `models/`.

### Шаг 3. Проверка графа

```bash
# из корня репозитория
python inspect_graph.py
```

---

## Конфигурация (`params.yaml`)

Пример структуры конфигурационного файла:

```yaml
experiment_name: 'test_exp'

preprocess:
  load_full_grid: true          # true — все ячейки активны; false — только NTG == 1
  use_labels: true              # использовать NTG как маску меток
  static_features: ['NTG', 'PORO', 'PERMX', 'PERMY', 'PERMZ']
  dynamic_features: ['SWAT', 'SOIL', 'SGAS']
  grid_features: ['GV', 'X', 'Y', 'Z']
  edge_feature_list: ['TRAN', 'DIST']
  scaler_type: 'stand'          # 'stand' | 'norm' | null
  max_workers: 16
  multiply_features: false

model:
  nz: 96                        # hidden_dim
  num_layers: 3
  dropout: 0.1
  ffn_expansion: 4

train:
  train_size: 0.6
  batch_size: 2
  epochs: 192
  warmup_learning_rate: 1.5e-5
  weight_decay: 7.4e-6
  seed: 42
  log_every: 10
  validate_every: 10
  patience: 4

paths:
  raw_data: /path/to/GNN_dataset_test/raw
  processed_data: /path/to/GNN_dataset_test/processed
  models: models
  checkpoints: checkpoints
```

---

## Обучение моделей

| Скрипт | Модель | Описание |
|--------|--------|----------|
| `train_gnn_plus.py` | `GNNPlusHetero` (v2) | **Основной** — TransformerConv + FFN-блоки GNN+ |
| `train.py` | `SimpleHeteroGNN` | Базовая GAT-архитектура |
| `train_sageconv.py` | `GNNPlusHeteroSAGE` | Вариант на GraphSAGE с edge features |

Загрузка данных, нормализация и разбиение train/val выполняются функцией `load_graph_data()` в `utils.py`:

- статистики для `cell.x` — из `processed/metadata/metadata.csv`;
- статистики для `well.y` и `edge_attr` — вычисляются по train-выборке;
- разбиение — `sklearn.model_selection.train_test_split` с фиксированным `seed`.

---

## Подбор гиперпараметров (Optuna)

В каталоге `optuna/` находятся скрипты для автоматического поиска гиперпараметров:

| Скрипт | Архитектура |
|--------|-------------|
| `optuna_transformerconv_no_prune.py` | TransformerConv |
| `optuna_search_gatv2.py` | GATv2Conv |
| `optuna_gcn_conv_edge.py` | GCN + edge features |
| `optuna_sage_conv_edge.py` | SAGE + edge features |
| `optuna_gin_conv_edge.py` | GIN + edge features |

Лучшие найденные параметры сохранены в `optuna/best_params_*.yaml`, например:

```yaml
hidden_dim: 64
num_layers: 3
dropout: 0.1
lr: 1.75e-5
weight_decay: 9.5e-6
```

Запуск (из `GNN_dataset_test/`):

```bash
python optuna/optuna_transformerconv_no_prune.py
```

---

## Визуализация и отладка

| Файл | Назначение |
|------|------------|
| `inspect_graph.py` | Проверка NaN/Inf, размерностей узлов, рёбер и целевых значений |
| `visualize_graph.py` | 2D-отрисовка подграфа ячеек через NetworkX |
| `visualize_3d.py` | 3D-визуализация сетки и свойств (xtgeo + pyvista) |
| `dataloaders.ipynb` | Исследование данных и загрузчиков |
| `test_model.py` | Быстрый smoke-тест `SimpleHeteroGNN` на одном `.pt`-файле |

Перед запуском скриптов визуализации обновите пути к `.pt` / `.grdecl` в начале файлов.

---

## Зависимости

| Пакет | Назначение |
|-------|------------|
| `torch`, `torch-geometric` | GNN-модели и графовые структуры |
| `resdata`, `resfo` | Чтение ECLIPSE SMSPEC / UNRST / UNSMRY |
| `xtgeo`, `pyvista` | Работа с GRDECL-сетками и 3D-визуализация |
| `numpy`, `pandas` | Обработка числовых данных |
| `scikit-learn` | Разбиение train/val |
| `optuna` | Подбор гиперпараметров |
| `PyYAML` | Конфигурация экспериментов |
| `matplotlib`, `networkx` | Визуализация |

---

## Примечания

- Проницаемости `PERMX`, `PERMY`, `PERMZ` логарифмируются (`log1p`) при препроцессинге.
- Целевые дебиты также проходят через `log1p` перед обучением.
- Первый временной шаг в `well.y` отбрасывается — модель предсказывает шаги 2–25 (итого 24).
- Для воспроизводимости используется фиксированный `seed` в `params.yaml`.
