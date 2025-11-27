# MMOE.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """ 单个专家网络 """
    def __init__(self, input_feature_size, expert_hidden_size, expert_output_size):
        super(Expert, self).__init__()
        # 第一层：输入 -> 隐藏
        self.fc1 = nn.Linear(input_feature_size, expert_hidden_size)
        # 第二层：隐藏 -> 输出
        self.fc2 = nn.Linear(expert_hidden_size, expert_output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: [batch_size, input_feature_size]
        out = self.fc1(x)       # [batch_size, expert_hidden_size]
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)     # [batch_size, expert_output_size]
        return out

class Tower(nn.Module):
    """
    每个任务对应的 Tower（塔网络）

    典型用法：接收该任务对应的 CGC 输出（已是任务级的表征），
    再经过两层 MLP + Sigmoid 得到最终的二分类概率。
    """
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        # 第一层：输入 -> 隐藏
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 第二层：隐藏 -> 输出（一般为 1 维，做二分类）
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, input_size]
        out = self.fc1(x)       # [batch_size, hidden_size]
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)     # [batch_size, output_size]
        out = self.sigmoid(out) # [batch_size, output_size] 概率
        return out


class MMOE(nn.Module):
    """
    标准 MMoE（Multi-gate Mixture of Experts）
    - 一个共享输入 x_full
    - K 个共享专家
    - 每个任务一个 gate
    - 每个任务一个 tower
    """
    def __init__(self,
                 input_size_full,
                 emb_dim,
                 num_experts=8,
                 expert_hidden_dim=32,
                 expert_output_dim=32,
                 tower_hidden_dim=8):
        super().__init__()
        self.emb_size = emb_dim
        self.emb = nn.Linear(input_size_full, emb_dim)
        self.num_experts = num_experts
        self.expert_output_dim = expert_output_dim
        self.tower_hidden_dim = tower_hidden_dim
        # ---- Shared Experts ----
        self.experts = nn.ModuleList([
            Expert(emb_dim, expert_hidden_dim, expert_output_dim)
            for _ in range(num_experts)
        ])

        # ---- Gates for each task ----
        self.gate_task1 = nn.Sequential(
            nn.Linear(emb_dim, num_experts),
            nn.Softmax(dim=1)
        )
        self.gate_task2 = nn.Sequential(
            nn.Linear(emb_dim, num_experts),
            nn.Softmax(dim=1)
        )

        # ---- Task-specific Towers ----
        self.tower1 = Tower(self.expert_output_dim, 1, self.tower_hidden_dim)
        self.tower2 = Tower(self.expert_output_dim, 1, self.tower_hidden_dim)

    def forward(self, x_full):
        """
        输入：
          - x_full: (batch, input_size_full)
        """

        x_full = self.emb(x_full)
        # ---- Expert outputs ----
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x_full))  # (batch, expert_dim)

        # convert to (num_experts, batch, dim)
        expert_outputs = torch.stack(expert_outputs, dim=0)

        # ---- Task1 Gate ----
        gate1 = self.gate_task1(x_full)          # (batch, num_experts)
        task1_out = torch.einsum("ebd,be->bd", expert_outputs, gate1)

        # ---- Task2 Gate ----
        gate2 = self.gate_task2(x_full)
        task2_out = torch.einsum("ebd,be->bd", expert_outputs, gate2)

        # ---- Task Towers ----
        y1 = self.tower1(task1_out)
        y2 = self.tower2(task2_out)

        return [y1, y2]
