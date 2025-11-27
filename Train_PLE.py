import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import pandas as pd
from sklearn.metrics import roc_auc_score
import csv
from Models_PLE import Expert, Tower, CGC, PLE


def to_categorical(y, num_classes=None, dtype='float32'):
    """
    将类别标签 y 转换为 one-hot 编码（类似 Keras 的 to_categorical）

    参数：
    - y: 一维或多维整数标签
    - num_classes: 类别数，如果不指定则按 y 中最大值+1 推断
    - dtype: 结果数据类型

    返回：
    - categorical: 与 y 形状基本一致、最后一维变成 num_classes 的 one-hot 矩阵
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    # 如果输入是 (..., 1) 形状，则去掉最后一个维度
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    # 展平成一维
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    # 初始化全 0 的 one-hot 矩阵
    categorical = np.zeros((n, num_classes), dtype=dtype)
    # 行索引为样本，列索引为类别，将对应位置置 1
    categorical[np.arange(n), y] = 1
    # 恢复到原来的高维结构，只是在最后增加 one-hot 维度
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def data_preparation():
    """
    读取 census-income 数据，做特征处理和数据集划分。

    返回：
    - train_data: 训练特征 DataFrame
    - train_label: [income_labels, marital_labels]（one-hot）
    - validation_data: 验证特征 DataFrame
    - validation_label: 同上（对 test 集随机一半）
    - test_data: 测试特征 DataFrame
    - test_label: 同上（剩下一半）
    - output_info: 各任务输出维度和名称（这里两个任务都是 2 类）
    """
    # 原始数据的列名
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']

    # 读取训练集 / 测试集（原作者将“test”再划一半做 valid）
    train_df = pd.read_csv('census-income.data.gz', delimiter=',', header=None,
                           index_col=None, names=column_names)
    test_df = pd.read_csv('census-income.test.gz', delimiter=',', header=None,
                          index_col=None, names=column_names)

    # 多任务标签列：收入是否 >50K & 婚姻状态
    label_columns = ['income_50k', 'marital_stat']

    # 需要做 one-hot 的类别特征列
    categorical_columns = [
        'class_worker', 'det_ind_code', 'det_occ_code', 'education',
        'hs_college', 'major_ind_code', 'major_occ_code', 'race',
        'hisp_origin', 'sex', 'union_member', 'unemp_reason',
        'full_or_part_emp', 'tax_filer_stat', 'region_prev_res',
        'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
        'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same',
        'mig_prev_sunbelt', 'fam_under_18', 'country_father',
        'country_mother', 'country_self', 'citizenship',
        'vet_question'
    ]

    # 对训练集与测试集分别做 one-hot（注意列可能不完全一致）
    train_transformed = pd.get_dummies(
        train_df.drop(label_columns, axis=1),
        columns=categorical_columns
    )
    test_transformed = pd.get_dummies(
        test_df.drop(label_columns, axis=1),
        columns=categorical_columns
    )

    # 原始标签 DataFrame
    train_labels = train_df[label_columns]
    test_labels = test_df[label_columns]

    # 手工补齐测试集中缺失的一列 dummy，使其与训练集列对齐
    # （这个数据集里，训练集中有该取值，测试集中没有出现）
    test_transformed['det_hh_fam_stat_ Grandchild <18 ever marr not in subfamily'] = 0

    # ---------- 构造两个任务的二分类 one-hot 标签 ----------
    # 任务1：收入是否 > 50000
    train_income = to_categorical(
        (train_labels.income_50k == ' 50000+.').astype(int),
        num_classes=2
    )
    # 任务2：婚姻状态是否“Never married”
    train_marital = to_categorical(
        (train_labels.marital_stat == ' Never married').astype(int),
        num_classes=2
    )
    other_income = to_categorical(
        (test_labels.income_50k == ' 50000+.').astype(int),
        num_classes=2
    )
    other_marital = to_categorical(
        (test_labels.marital_stat == ' Never married').astype(int),
        num_classes=2
    )

    # 方便后续扩展任务，用 dict 存任务名 → 输出维度 / 标签
    dict_outputs = {
        'income': train_income.shape[1],
        'marital': train_marital.shape[1]
    }
    dict_train_labels = {
        'income': train_income,
        'marital': train_marital
    }
    dict_other_labels = {
        'income': other_income,
        'marital': other_marital
    }
    # output_info: [(2, 'income'), (2, 'marital')]
    output_info = [(dict_outputs[key], key) for key in sorted(dict_outputs.keys())]

    # ---------- 将官方 test 集再划分为 valid / test ----------
    # 按随机种子从 test_transformed 抽 50% 做验证
    validation_indices = test_transformed.sample(
        frac=0.5,
        replace=False,
        random_state=seed
    ).index
    # 剩余的一半做最终测试
    test_indices = list(set(test_transformed.index) - set(validation_indices))

    validation_data = test_transformed.iloc[validation_indices]
    validation_label = [
        dict_other_labels[key][validation_indices]
        for key in sorted(dict_other_labels.keys())
    ]
    test_data = test_transformed.iloc[test_indices]
    test_label = [
        dict_other_labels[key][test_indices]
        for key in sorted(dict_other_labels.keys())
    ]

    # 训练数据用整个官方 train 集
    train_data = train_transformed
    train_label = [
        dict_train_labels[key]
        for key in sorted(dict_train_labels.keys())
    ]

    return train_data, train_label, validation_data, validation_label, test_data, test_label, output_info


def getTensorDataset(my_x, my_y):
    """
    将 numpy / DataFrame 格式的特征和标签包装成 TensorDataset，
    方便后续用 DataLoader 做批训练。
    """
    # 特征转成 float32 Tensor
    tensor_x = torch.Tensor(my_x.astype(np.float32))
    # 标签直接转成 Tensor（此处是 0/1 整数）
    tensor_y = torch.Tensor(my_y)
    # 返回 (x, y) 组成的 TensorDataset
    return torch.utils.data.TensorDataset(tensor_x, tensor_y)


def test(loader):
    """
    在给定 DataLoader 上评估模型，返回两个任务的 AUC。

    注意：
    - 使用的是全局的 model / device / loss_fn
    """
    t1_pred, t2_pred, t1_target, t2_target = [], [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # 前向计算，得到两个任务的预测
            yhat = model(x)
            y1, y2 = y[:, 0], y[:, 1]      # 真实标签 (batch,)
            yhat_1, yhat_2 = yhat[0], yhat[1]  # 预测概率 (batch, 1)

            # 计算两个任务的 BCE 损失（这里只是演示，没有返回）
            loss1 = loss_fn(yhat_1, y1.view(-1, 1))
            loss2 = loss_fn(yhat_2, y2.view(-1, 1))
            loss = loss1 + loss2  # 未使用，仅可用于调试

            # --------- 收集预测值和真实标签用于 AUC 计算 ---------
            # yhat_x: (batch, 1) → 展平到 (batch,)
            t1_hat = yhat_1.cpu().numpy().ravel()
            t2_hat = yhat_2.cpu().numpy().ravel()

            # 这里用 extend 而不是 += list(...)，保证最后是一维数组
            t1_pred.extend(t1_hat.tolist())
            t2_pred.extend(t2_hat.tolist())
            t1_target.extend(y1.cpu().numpy().ravel().tolist())
            t2_target.extend(y2.cpu().numpy().ravel().tolist())

    # 计算两个任务的 AUC（sklearn 会将 list 转成一维 numpy）
    auc_1 = roc_auc_score(t1_target, t1_pred)
    auc_2 = roc_auc_score(t2_target, t2_pred)
    return auc_1, auc_2


# ------------------- 训练主程序部分 -------------------

# 固定随机种子，保证结果可复现（还可以加 torch.manual_seed 等）
random.seed(3)
np.random.seed(3)
seed = 3
batch_size = 1024

# 自动选择 GPU / CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据准备
train_data, train_label, validation_data, validation_label, \
    test_data, test_label, output_info = data_preparation()

# 原始标签是 one-hot，这里通过 argmax 转回 0/1
# 然后将两个任务的标签拼成 (N, 2) 的二维数组
train_label_tmp = np.column_stack((
    np.argmax(train_label[0], axis=1),
    np.argmax(train_label[1], axis=1)
))
train_loader = DataLoader(
    dataset=getTensorDataset(train_data.to_numpy(), train_label_tmp),
    batch_size=batch_size,
    shuffle=True  # 训练集打乱
)

validation_label_tmp = np.column_stack((
    np.argmax(validation_label[0], axis=1),
    np.argmax(validation_label[1], axis=1)
))
val_loader = DataLoader(
    dataset=getTensorDataset(validation_data.to_numpy(), validation_label_tmp),
    batch_size=batch_size
)

test_label_tmp = np.column_stack((
    np.argmax(test_label[0], axis=1),
    np.argmax(test_label[1], axis=1)
))
test_loader = DataLoader(
    dataset=getTensorDataset(test_data.to_numpy(), test_label_tmp),
    batch_size=batch_size
)

# 构建 PLE 模型
# input_size 必须与 train_data 的特征列数一致（此处是 499）
model = PLE(
    num_CGC_layers=4,
    input_size=499,
    emb_dim = 128,
    num_specific_experts=4,
    num_shared_experts=4,
    experts_out=32,
    experts_hidden=32,
    towers_hidden=8
)
model = model.to(device)

# 优化器、损失函数等训练超参
lr = 1e-4
n_epochs = 100
# 每个任务是二分类，输出层已 sigmoid，因此用 BCELoss
loss_fn = nn.BCELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

losses = []    # 记录训练集 loss 曲线
val_loss = []  # 代码里未真正使用，可按需扩展

# 训练过程，同时把每个 epoch 的指标写入 CSV 方便画图 / 对比
with open("PLE_results.csv", "w", newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # CSV 表头
    csvwriter.writerow(["Epoch", "Train Loss", "Val Task1 AUC", "Val Task2 AUC"])

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = []
        print("Epoch: {}/{}".format(epoch, n_epochs))

        # --------- 一个 epoch 的 mini-batch 训练 ---------
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)         # 前向：得到两个任务的预测

            y1, y2 = y[:, 0], y[:, 1]    # 两个任务的真实标签
            y_1, y_2 = y_hat[0], y_hat[1]

            # 分别计算两个任务的 BCE
            loss1 = loss_fn(y_1, y1.view(-1, 1))
            loss2 = loss_fn(y_2, y2.view(-1, 1))
            loss = loss1 + loss2       # 简单相加作为总多任务损失

            loss.backward()            # 反向传播
            optimizer.step()           # 参数更新
            optimizer.zero_grad()      # 清空梯度

            epoch_loss.append(loss.item())

        # 记录并打印本 epoch 的平均训练损失
        mean_train_loss = np.mean(epoch_loss)
        losses.append(mean_train_loss)

        # 在验证集上评估两个任务 AUC
        auc1, auc2 = test(val_loader)
        print(
            'train loss: {:.5f}, val task1 auc: {:.5f}, val task2 auc: {:.3f}'
            .format(mean_train_loss, auc1, auc2)
        )
        # 写入 CSV 文件
        csvwriter.writerow([epoch, mean_train_loss, auc1, auc2])

    # --------- 训练结束后，在测试集上做最终评估 ---------
    auc1, auc2 = test(test_loader)
    print('PLE: test task1 auc: {:.3f}, test task2 auc: {:.3f}'.format(auc1, auc2))

    # 保存训练好的模型参数（state_dict）
    torch.save(model.state_dict(), "model_ple.pth")
