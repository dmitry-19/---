# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from src.model import BaseCallerNet
from src.dataset_loader import ChromDataset
from src.logger import setup_logger

logger = setup_logger("train", log_file="training.log")

def train():
    # Параметры
    BATCH_SIZE = 64
    LR = 1e-4
    EPOCHS = 50
    PATIENCE = 10  # для early stopping
    DATA_DIR = "datasets"

    # Загрузка полного датасета
    dataset = ChromDataset(
        os.path.join(DATA_DIR, "X.npy"),
        os.path.join(DATA_DIR, "y.npy"),
        os.path.join(DATA_DIR, "meta.npy")
    )

    # Разделение на train/val/test (80/10/10)
    total = len(dataset)
    train_len = int(0.8 * total)
    val_len = int(0.1 * total)
    test_len = total - train_len - val_len
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])

    # Вычисление весов классов (из train_ds)
    all_labels = [dataset[i][1].item() for i in range(len(dataset))]
    # Используем train_ds индексы
    train_labels = [all_labels[i] for i in train_ds.indices]
    _, counts = np.unique(train_labels, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)  # нормализация
    class_weights = torch.tensor(weights, dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Модель
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaseCallerNet(use_meta=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            if len(batch) == 3:
                x, y, meta = batch
                x, y, meta = x.to(device), y.to(device), meta.to(device)
                out = model(x, meta)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                out = model(x)

            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            pred = out.argmax(dim=1)
            train_correct += (pred == y).sum().item()
            train_total += len(y)

        train_acc = train_correct / train_total

        # Валидация
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    x, y, meta = batch
                    x, y, meta = x.to(device), y.to(device), meta.to(device)
                    out = model(x, meta)
                else:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    out = model(x)

                pred = out.argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += len(y)
        val_acc = val_correct / val_total

        logger.info(f"Epoch {epoch:2d}: Train loss={train_loss/train_total:.4f}, Train acc={train_acc:.4f}, Val acc={val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), "basecaller.pt")
            logger.info("  => saved best model")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info(f"Training finished. Best model at epoch {best_epoch} with val acc {best_val_acc:.4f}")

    # Тест
    model.load_state_dict(torch.load("basecaller.pt"))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x, y, meta = batch
                x, y, meta = x.to(device), y.to(device), meta.to(device)
                out = model(x, meta)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                out = model(x)
            pred = out.argmax(dim=1)
            test_correct += (pred == y).sum().item()
            test_total += len(y)
    test_acc = test_correct / test_total
    logger.info(f"Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    train()