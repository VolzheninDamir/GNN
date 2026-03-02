import torch
import yaml
import os
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

# Импортируем твою модель и функции
from simple_model import SimpleHeteroGNN
from utils import load_graph_data

def train_model():
    # 1. Загрузка параметров из yaml
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Работаем на: {device}")

    # 2. Загрузка данных через твою функцию
    # Она сама сделает split, посчитает статистики и нормализует всё
    train_loader, val_loader, feat_list = load_graph_data(
        config['paths'], 
        config['preprocess'], 
        config['train']
    )

    # 3. Инициализация модели
    # Берем один пример из базы, чтобы узнать размерности входных данных
    sample_data, _ = next(iter(train_loader))
    
    model = SimpleHeteroGNN(
        cell_features=sample_data['cell'].x.size(1),
        well_features=sample_data['well'].x.size(1),
        hidden_dim=config['model']['nz'], # используем nz из yaml
        out_seq_len=25,
        num_phases=3
    ).to(device)

    # 4. Настройка обучения
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['train']['warmup_learning_rate'],
        weight_decay=config['train']['weight_decay']
    )
    criterion = torch.nn.MSELoss()

    # 5. Цикл обучения
    epochs = config['train']['epochs']
    train_history = []
    val_history = []

    print(f"Начинаем обучение на {len(train_loader.dataset)} графах...")

    for epoch in range(1, epochs + 1):
        # --- ТРЕНИРОВКА ---
        model.train()
        total_train_loss = 0
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred = model(batch)

            #добавлено новое
            if torch.isnan(batch['well'].y).any():
                print("NaN в well.y!")
            if torch.isnan(batch['cell'].x).any():
                print("NaN в cell.x!")
            if torch.isnan(pred).any():
                print("NaN в предсказании!")

            #добавлено новое
            print(f"Batch well.y: min={batch['well'].y.min():.3f}, max={batch['well'].y.max():.3f}, mean={batch['well'].y.mean():.3f}")
            # ВАЖНО: твой Dataset уже нормализовал well.y, так что считаем лосс прямо так
            loss = criterion(pred, batch['well'].y)
            
            loss.backward()
            #добавлено новое
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_history.append(avg_train_loss)

        # --- ВАЛИДАЦИЯ ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for v_batch, _ in val_loader:
                v_batch = v_batch.to(device)
                v_pred = model(v_batch)
                v_loss = criterion(v_pred, v_batch['well'].y)
                total_val_loss += v_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_history.append(avg_val_loss)

        if epoch % 1 == 0:
            print(f"Эпоха {epoch:03d} | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")

    # 6. Сохранение результата
    os.makedirs(config['paths']['models'], exist_ok=True)
    save_path = os.path.join(config['paths']['models'], 'gnn_oil_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Модель сохранена в {save_path}")

    # 7. График лосса
    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label='Train Loss')
    plt.plot(val_history, label='Val Loss')
    plt.yscale('log')
    plt.xlabel('Эпоха')
    plt.ylabel('MSE (нормализованный)')
    plt.legend()
    plt.title('Процесс обучения')
    plt.show()

if __name__ == "__main__":
    # Фикс для корректной работы multiprocessing в Windows
    train_model()