import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.metrics import roc_auc_score
import csv

# 导入 PLE / AOPE 模型
from Models_PLE import PLE
from Models_HFS_OPE import HFSOPE   

# 从 PLE 训练脚本中复用一些公共工具和全局变量
from Train_PLE import (
    device,           # 设备 'cuda' or 'cpu'
    test,             # 用于评估 PLE 的 test(loader) 函数
    batch_size,       # 批大小
    data_preparation, # 数据预处理函数
    getTensorDataset, # 将 numpy/DataFrame + label 封装为 TensorDataset 的函数
    val_loader,       # PLE 使用的验证集 DataLoader（one-hot 全特征 + 标签）
    train_label_tmp,  # (N, 2) 训练标签（两个任务）
    validation_label_tmp,
    test_label_tmp
)


# =======================
# 一、特征重要性相关函数
# =======================

def map_features_to_columns(train_df, validation_data, label_columns, categorical_columns):
    """
    构造“原始特征名 → One-Hot 后所有对应列名”的映射字典。
    """
    original_features = list(train_df.drop(label_columns, axis=1).columns)
    feature_to_columns = {}
    for feature in original_features:
        if feature in categorical_columns:
            cols = [col for col in validation_data.columns if col.startswith(feature + "_")]
            if cols:
                feature_to_columns[feature] = cols
        else:
            if feature in validation_data.columns:
                feature_to_columns[feature] = [feature]
    return feature_to_columns


def generate_masked_validation_data(validation_data, feature_to_columns, categorical_columns):
    """
    生成“掩盖单个原始特征”的验证集列表，用于后续做 AUC-drop 特征重要性。
    """
    masked_validation_data_list = []
    for feature, cols in feature_to_columns.items():
        masked_validation = validation_data.copy()
        if feature not in categorical_columns:
            masked_validation[cols] = masked_validation[cols[0]].mean()
        else:
            masked_validation[cols] = 0
        masked_validation_data_list.append((feature, masked_validation))
    
    return masked_validation_data_list


def compute_feature_importance(model_path, val_loader, masked_validation_data_list,
                               validation_label_tmp, batch_size, device):
    """
    使用训练好的 PLE 模型，基于“特征掩蔽 → AUC 降幅”来计算特征重要性。
    """
    # 1）初始化 PLE 模型
    model = PLE(
        num_CGC_layers=4,
        input_size=499, 
        emb_dim=128,                 # 若特征数变化，可改为 train_data.shape[1]
        num_specific_experts=4,
        num_shared_experts=4,
        experts_out=32,
        experts_hidden=32,
        towers_hidden=8
    )
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 2）原始验证集 AUC（基线）
    baseline_auc1, baseline_auc2 = test(val_loader)

    # 3）逐特征掩蔽并计算 AUC drop
    importance_results = []
    for feature, masked_validation in masked_validation_data_list:
        masked_loader = DataLoader(
            dataset=getTensorDataset(masked_validation.to_numpy(), validation_label_tmp),
            batch_size=batch_size
        )
        auc1_masked, auc2_masked = test(masked_loader)
        importance_results.append({
            "Feature": feature,
            "Importance1": baseline_auc1 - auc1_masked,
            "Importance2": baseline_auc2 - auc2_masked
        })
    
    return importance_results


def get_top_features(importance_results, top_n=5):
    """
    分别取出两个任务的前 top_n 个重要特征。
    """
    importance_df = pd.DataFrame(importance_results)

    top_task1 = importance_df.sort_values(by="Importance1", ascending=False).head(top_n)
    top_task2 = importance_df.sort_values(by="Importance2", ascending=False).head(top_n)
    
    top_features_task1 = top_task1["Feature"].tolist()
    top_features_task2 = top_task2["Feature"].tolist()

    return top_features_task1, top_features_task2


def extract_features_from_dataframe(df, top_features, feature_to_columns):
    """
    根据“原始特征名列表”从 one-hot 后的 DataFrame 中提取对应的列。
    """
    extracted_df = pd.DataFrame()
    for feature in top_features:
        if feature in feature_to_columns:
            extracted_df = pd.concat([extracted_df, df[feature_to_columns[feature]]], axis=1)
        else:
            print(f"Warning: {feature} 不在映射中！")
    return extracted_df


def create_dataloader(full_data, task1_data, task2_data, labels, batch_size, shuffle):
    """
    构造适用于 OPE/AOPE 的 DataLoader：
    """
    tensor_full = torch.Tensor(full_data.to_numpy().astype(np.float32))
    tensor_task1 = torch.Tensor(task1_data.to_numpy().astype(np.float32))
    tensor_task2 = torch.Tensor(task2_data.to_numpy().astype(np.float32))
    tensor_y = torch.Tensor(labels)

    dataset = TensorDataset(tensor_full, tensor_task1, tensor_task2, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def test_ope(model, loader):
    """
    在 OPE/AOPE 模型上评估两个任务的 AUC。
    """
    t1_pred, t2_pred, t1_target, t2_target = [], [], [], []
    model.eval()
    with torch.no_grad():
        for x, x1, x2, y in loader:
            x, x1, x2, y = x.to(device), x1.to(device), x2.to(device), y.to(device)
            yhat = model(x, x1, x2)  # [yhat_task1, yhat_task2]
            y1, y2 = y[:, 0], y[:, 1]
            yhat_1, yhat_2 = yhat[0], yhat[1]

            # loss 仅用于调试
            _ = loss_fn(yhat_1, y1.view(-1, 1)) + loss_fn(yhat_2, y2.view(-1, 1))

            t1_pred.extend(yhat_1.cpu().numpy().ravel().tolist())
            t2_pred.extend(yhat_2.cpu().numpy().ravel().tolist())
            t1_target.extend(y1.cpu().numpy().ravel().tolist())
            t2_target.extend(y2.cpu().numpy().ravel().tolist())

    auc_1 = roc_auc_score(t1_target, t1_pred)
    auc_2 = roc_auc_score(t2_target, t2_pred)
    return auc_1, auc_2


# =======================
# 二、特征重要性提取 + 构造任务私有特征
# =======================

# 1）原始列名（与 Train_PLE 中一致）
column_names = [
    'age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education',
    'wage_per_hour', 'hs_college', 'marital_stat', 'major_ind_code',
    'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses',
    'stock_dividends', 'tax_filer_stat', 'region_prev_res', 'state_prev_res',
    'det_hh_fam_stat', 'det_hh_summ', 'instance_weight', 'mig_chg_msa',
    'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
    'num_emp', 'fam_under_18', 'country_father', 'country_mother',
    'country_self', 'citizenship', 'own_or_self', 'vet_question',
    'vet_benefits', 'weeks_worked', 'year', 'income_50k'
]

# 标签列
label_columns = ['income_50k', 'marital_stat']

# 需要 one-hot 的类别特征列
categorical_columns = [
    'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college',
    'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex',
    'union_member', 'unemp_reason', 'full_or_part_emp', 'tax_filer_stat',
    'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
    'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same',
    'mig_prev_sunbelt', 'fam_under_18', 'country_father', 'country_mother',
    'country_self', 'citizenship', 'vet_question'
]

# 2）读取原始 train / test
train_df = pd.read_csv('census-income.data.gz', delimiter=',', header=None,
                       index_col=None, names=column_names)
test_df = pd.read_csv('census-income.test.gz', delimiter=',', header=None,
                      index_col=None, names=column_names)

# 3）调用 data_preparation()（来自 Train_PLE），获得 one-hot 后的 train/val/test
train_data, train_label, validation_data, validation_label, \
    test_data, test_label, output_info = data_preparation()

# 4）构造 原始特征 → one-hot 后列名 的映射
feature_to_columns = map_features_to_columns(
    train_df, validation_data, label_columns, categorical_columns
)

# 5）对验证集做“掩蔽某个原始特征”
masked_validation_data_list = generate_masked_validation_data(
    validation_data, feature_to_columns, categorical_columns
)

# 6）使用训练好的 PLE 模型计算 “两个任务上的特征重要性”
importance_results = compute_feature_importance(
    model_path="model_ple.pth",
    val_loader=val_loader,
    masked_validation_data_list=masked_validation_data_list,
    validation_label_tmp=validation_label_tmp,
    batch_size=batch_size,
    device=device
)

# 7）分别取出两个任务的 top-K 重要特征列表
top_features_task1, top_features_task2 = get_top_features(
    importance_results, top_n=5
)

# 8）针对 train/val/test 三个数据集，抽取对应“任务私有特征子集”

# --- 训练集 ---
task1_train_data = extract_features_from_dataframe(
    train_data, top_features_task1, feature_to_columns
)
task2_train_data = extract_features_from_dataframe(
    train_data, top_features_task2, feature_to_columns
)
train_loader = create_dataloader(
    train_data,
    task1_train_data,
    task2_train_data,
    train_label_tmp,
    batch_size,
    shuffle=True
)

# --- 验证集 ---
task1_val_data = extract_features_from_dataframe(
    validation_data, top_features_task1, feature_to_columns
)
task2_val_data = extract_features_from_dataframe(
    validation_data, top_features_task2, feature_to_columns
)
val_loader = create_dataloader(
    validation_data,
    task1_val_data,
    task2_val_data,
    validation_label_tmp,
    batch_size,
    shuffle=False
)

# --- 测试集 ---
task1_test_data = extract_features_from_dataframe(
    test_data, top_features_task1, feature_to_columns
)
task2_test_data = extract_features_from_dataframe(
    test_data, top_features_task2, feature_to_columns
)
test_loader = create_dataloader(
    test_data,
    task1_test_data,
    task2_test_data,
    test_label_tmp,
    batch_size,
    shuffle=False
)


# =======================
# 三、HFSOPE 模型训练
# =======================

# 训练相关超参数
lr = 1e-4
n_epochs = 100
loss_fn = nn.BCELoss(reduction='mean')
lambda_gate = 1e-4   # 稀疏 gate 正则系数，可调或设为 0 关闭
optimizer = None

# 构建 AOPE 模型
model_aope = HFSOPE(
    input_size_full=train_data.shape[1],              
    input_size_task1=task1_train_data.shape[1],
    input_size_task2=task2_train_data.shape[1],
    emb_dim_full=128,
    emb_dim_task1=64,
    emb_dim_task2=64,
    num_CGC_layers=4,
    num_specific_experts=4,
    num_shared_experts=4,
    experts_out=32,
    experts_hidden=32,
    towers_hidden=8,
    gate_hidden_ratio=1.0                             # gate MLP 隐层比例，可调
)
model_aope = model_aope.to(device)

optimizer = optim.Adam(model_aope.parameters(), lr=lr, weight_decay=1e-5)
losses = []

# 将每个 epoch 的结果写入 CSV，方便对比 PLE / OPE / AOPE
with open("HFSOPE_results.csv", "w", newline='') as csvfile:   # ★ 文件名修改
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Epoch", "Train Loss", "Val Task1 AUC", "Val Task2 AUC"])

    for epoch in range(n_epochs):
        model_aope.train()
        epoch_loss = []
        print("Epoch: {}/{}".format(epoch, n_epochs))

        # --------- 一个 epoch 的 mini-batch 训练 ---------
        for x, x1, x2, y in train_loader:
            x, x1, x2, y = x.to(device), x1.to(device), x2.to(device), y.to(device)
            y_hat = model_aope(x, x1, x2)  # [yhat_task1, yhat_task2]
            y1, y2 = y[:, 0], y[:, 1]

            base_loss = loss_fn(y_hat[0], y1.view(-1, 1)) + \
                        loss_fn(y_hat[1], y2.view(-1, 1))

            # 稀疏 gate 正则（仅 AOPE 有）
            gate_reg = 0.0
            if model_aope.gate_task1.last_gate is not None and model_aope.gate_task2.last_gate is not None:
                gate_reg = (
                    model_aope.gate_task1.last_gate.abs().mean() +
                    model_aope.gate_task2.last_gate.abs().mean()
                )

            loss = base_loss + lambda_gate * gate_reg

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss.append(loss.item())

        avg_loss = np.mean(epoch_loss)
        losses.append(avg_loss)

        # 在验证集上评估两个任务的 AUC
        auc1, auc2 = test_ope(model_aope, val_loader)
        print(
            'Epoch {} - train loss: {:.5f}, val task1 auc: {:.5f}, val task2 auc: {:.5f}'
            .format(epoch, avg_loss, auc1, auc2)
        )
        csvwriter.writerow([epoch, avg_loss, auc1, auc2])

    # --------- 训练结束后在测试集评估 ---------
    auc1, auc2 = test_ope(model_aope, test_loader)
    print('HFS_OPE: Test Task1 AUC: {:.3f}, Test Task2 AUC: {:.3f}'.format(auc1, auc2))

    # 保存 HFSOPE 模型参数
    torch.save(model_aope.state_dict(), "model_hfsope.pth")   
