# Train_MMOE.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import csv

from Train_PLE import (
    device, data_preparation,
    batch_size, train_label_tmp,
    validation_label_tmp, test_label_tmp,
    getTensorDataset
)

from Models_MMOE import MMOE
from sklearn.metrics import roc_auc_score


# ======================
# 加载数据
# ======================
train_data, train_label, validation_data, validation_label, test_data, test_label, output_info = data_preparation()

train_loader = DataLoader(
    dataset=getTensorDataset(train_data.to_numpy(), train_label_tmp),
    batch_size=batch_size,
    shuffle=True  # 训练集打乱
)

val_loader = DataLoader(
    dataset=getTensorDataset(validation_data.to_numpy(), validation_label_tmp),
    batch_size=batch_size
)

test_loader = DataLoader(
    dataset=getTensorDataset(test_data.to_numpy(), test_label_tmp),
    batch_size=batch_size
)


# ======================
# 构建 MMoE 模型
# ======================
model = MMOE(
    input_size_full=train_data.shape[1],
    num_experts=8,
    emb_dim=128,
    expert_hidden_dim=32,
    expert_output_dim=32,
    tower_hidden_dim=16
).to(device)

lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
loss_fn = nn.BCELoss(reduction='mean')


# ======================
# 测试函数
# ======================
def test_mmoe(model, loader):
    t1_pred, t2_pred, t1_true, t2_true = [], [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            yhat1, yhat2 = model(x)
            t1_true += list(y[:, 0].cpu().numpy())
            t2_true += list(y[:, 1].cpu().numpy())
            t1_pred += list(yhat1.cpu().numpy())
            t2_pred += list(yhat2.cpu().numpy())

    auc1 = roc_auc_score(t1_true, t1_pred)
    auc2 = roc_auc_score(t2_true, t2_pred)
    return auc1, auc2


# ======================
# 训练循环
# ======================
with open("MMOE_results.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "TrainLoss", "ValAUC1", "ValAUC2"])

    for epoch in range(100):
        model.train()
        losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y1, y2 = model(x)

            t1, t2 = y[:, 0], y[:, 1]
            loss = loss_fn(y1, t1.view(-1, 1)) + loss_fn(y2, t2.view(-1, 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        val_auc1, val_auc2 = test_mmoe(model, val_loader)

        print(f"Epoch {epoch}, Loss={avg_loss:.4f}, ValAUC1={val_auc1:.4f}, ValAUC2={val_auc2:.4f}")
        writer.writerow([epoch, avg_loss, val_auc1, val_auc2])

    torch.save(model.state_dict(), "model_mmoe.pth")


# ======================
# 测试集
# ======================
test_auc1, test_auc2 = test_mmoe(model, test_loader)
print(f"MMOE: Test AUC1={test_auc1:.4f}, Test AUC2={test_auc2:.4f}")
