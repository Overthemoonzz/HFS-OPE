import numpy as np
import torch
import torch.nn as nn

class Expert(nn.Module):
    """
    单个 Expert 模块（两层全连接 + ReLU + Dropout）

    参数说明：
    - input_feature_size : 输入特征维度
    - expert_hidden_size : Expert 内部隐藏层维度
    - expert_output_size : Expert 输出维度（供后续 gate 加权和、再给 Tower 使用）
    """
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


class CGC(nn.Module):
    """
    CGC（Customized Gate Control）模块，实现 PLE 中的一层“专家 + 门控”结构。

    对于 2 个任务的情况：
    - 有 num_specific_experts 个任务特定专家（每个任务一组）
    - 有 num_shared_experts 个共享专家（两个任务共用）
    - 对每个任务都有一个 gate（门控网络），根据输入特征生成对各个专家的权重，
      再对所有专家输出做加权和，得到该任务的表征。

    参数：
    - input_size           : 当前层输入的特征维度（第一层为原始特征维度，后续层为专家输出维度）
    - num_specific_experts : 每个任务的“特定专家”数量
    - num_shared_experts   : “共享专家”数量
    - experts_out          : 每个专家的输出维度（也是下一层/塔的输入维度）
    - experts_hidden       : 每个专家内部隐藏层维度
    - if_last              : 是否为最后一层 CGC
                             - False: 返回 [shared_out, task1_out, task2_out]
                             - True : 返回 [task1_out, task2_out]
    """
    def __init__(self, input_size, num_specific_experts, num_shared_experts,
                 experts_out, experts_hidden, if_last):
        super(CGC, self).__init__()

        self.input_size = input_size                      # 输入特征维度
        self.num_specific_experts = num_specific_experts  # 每个任务的专家数量
        self.num_shared_experts = num_shared_experts      # 共享专家数量
        self.experts_out = experts_out                    # 专家输出维度
        self.experts_hidden = experts_hidden              # 专家隐藏层维度
        self.if_last = if_last                            # 是否最后一层

        # ====== 定义 Expert 组 ======
        # 注意：这里要确保 Expert 的 hidden / output 维度顺序正确
        # 正确应该是：hidden = experts_hidden, output = experts_out
        self.experts_shared = nn.ModuleList([
            Expert(self.input_size, self.experts_hidden, self.experts_out)
            for _ in range(self.num_shared_experts)
        ])  # 共享专家组

        self.experts_task1 = nn.ModuleList([
            Expert(self.input_size, self.experts_hidden, self.experts_out)
            for _ in range(self.num_specific_experts)
        ])  # 任务1专属专家组

        self.experts_task2 = nn.ModuleList([
            Expert(self.input_size, self.experts_hidden, self.experts_out)
            for _ in range(self.num_specific_experts)
        ])  # 任务2专属专家组

        # Softmax（如果要单独用）
        self.soft = nn.Softmax(dim=1)

        # ====== 定义 Gate（门控网络） ======
        # gate_shared 输出维度 = 所有专家数量之和
        #   = num_specific_experts * 2 + num_shared_experts
        self.gate_shared = nn.Sequential(
            nn.Linear(self.input_size,
                      self.num_specific_experts * 2 + self.num_shared_experts),
            nn.Softmax(dim=1)
        )

        # 每个任务的 gate 需要在“该任务专属专家 + 共享专家”之间分配权重
        # 输出维度 = num_specific_experts + num_shared_experts
        self.gate_task1 = nn.Sequential(
            nn.Linear(self.input_size,
                      self.num_specific_experts + self.num_shared_experts),
            nn.Softmax(dim=1)
        )
        self.gate_task2 = nn.Sequential(
            nn.Linear(self.input_size,
                      self.num_specific_experts + self.num_shared_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        x:
        - 如果 if_last=False: 传入 [inputs_shared, inputs_task1, inputs_task2]
        - 第一层 PLE: 这三个一般都是同一个原始输入
        - 后续层 PLE: 三个分别是上一层输出的 shared / task1 / task2
        """
        inputs_shared, inputs_task1, inputs_task2 = x
        # 三者 shape 一般都是 [batch_size, input_size]

        # ====== 先计算所有专家的输出 ======
        # 共享专家输出列表：长度 = num_shared_experts
        experts_shared_o = [e(inputs_shared) for e in self.experts_shared]
        # stack 之后 shape: [num_shared_experts, batch_size, experts_out]
        experts_shared_o = torch.stack(experts_shared_o)

        # 任务1的专属专家输出
        experts_task1_o = [e(inputs_task1) for e in self.experts_task1]
        # [num_specific_experts, batch_size, experts_out]
        experts_task1_o = torch.stack(experts_task1_o)

        # 任务2的专属专家输出
        experts_task2_o = [e(inputs_task2) for e in self.experts_task2]
        # [num_specific_experts, batch_size, experts_out]
        experts_task2_o = torch.stack(experts_task2_o)

        # ====== 任务1的 gate 加权 ======
        # selected_task1: [batch_size, num_specific_experts + num_shared_experts]
        selected_task1 = self.gate_task1(inputs_task1)

        # 拼接当前任务的专属专家 + 共享专家
        # gate_expert_output1: [num_specific_experts + num_shared_experts,
        #                       batch_size, experts_out]
        gate_expert_output1 = torch.cat((experts_task1_o, experts_shared_o), dim=0)

        # 使用爱因斯坦求和进行加权：
        # 'abc, ba -> bc'
        #   a: 专家索引
        #   b: batch 索引
        #   c: 特征维度
        # experts: [a, b, c]
        # gate   : [b, a]
        # 输出   : [b, c] = [batch_size, experts_out]
        gate_task1_out = torch.einsum('abc, ba -> bc',
                                      gate_expert_output1, selected_task1)

        # ====== 任务2的 gate 加权（同理） ======
        selected_task2 = self.gate_task2(inputs_task2)  # [batch_size, num_specific_experts + num_shared_experts]
        gate_expert_output2 = torch.cat((experts_task2_o, experts_shared_o), dim=0)
        gate_task2_out = torch.einsum('abc, ba -> bc',
                                      gate_expert_output2, selected_task2)

        # ====== 共享门控，用于生成“共享表示”，供下一层使用 ======
        # selected_shared: [batch_size, num_specific_experts * 2 + num_shared_experts]
        selected_shared = self.gate_shared(inputs_shared)

        # 拼接：任务1专属 + 任务2专属 + 共享专家
        gate_expert_outputshared = torch.cat(
            (experts_task1_o, experts_task2_o, experts_shared_o),
            dim=0
        )  # [num_specific_experts*2 + num_shared_experts, batch_size, experts_out]

        gate_shared_out = torch.einsum('abc, ba -> bc',
                                       gate_expert_outputshared, selected_shared)
        # [batch_size, experts_out]

        # ====== 返回结果 ======
        # - 如果是中间层：返回三个输出，供下一层 CGC 使用
        # - 如果是最后一层：只返回两个任务输出，供 Tower 使用
        if self.if_last:
            return [gate_task1_out, gate_task2_out]
        else:
            return [gate_shared_out, gate_task1_out, gate_task2_out]


class PLE(nn.Module):
    """
    PLE（Progressive Layered Extraction）模型（两任务版本）

    结构：
    - 首先一层 CGC（cgc_layer1），输入是原始特征 x（shared/task1/task2 都为 x）
    - 随后再叠加 num_CGC_layers 层 CGC（cgc_layers），实现“渐进式抽取”
    - 最后一层 CGC 输出两个任务的最终表示，再分别送入对应的 Tower，得到两个任务的预测结果
    """
    def __init__(self, num_CGC_layers, input_size, emb_dim,
                 num_specific_experts, num_shared_experts,
                 experts_out, experts_hidden, towers_hidden):
        super(PLE, self).__init__()
        self.emb_size = emb_dim
        self.emb = nn.Linear(input_size, emb_dim)
        self.num_CGC_layers = num_CGC_layers
        self.input_size = input_size
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden

        # 第 1 层 CGC：输入是原始特征向量 x
        self.cgc_layer1 = CGC(
            self.emb_size,
            self.num_specific_experts,
            self.num_shared_experts,
            self.experts_out,
            self.experts_hidden,
            if_last=False
        )

        # 后续 CGC 层：输入为上一层专家输出（维度 = experts_out）
        # 注意：这里不要再写死 32，而是用 self.experts_out
        self.cgc_layers = nn.ModuleList([
            CGC(
                self.experts_out,            # 输入维度 = 专家输出维度
                num_specific_experts,
                num_shared_experts,
                experts_out,
                experts_hidden,
                if_last=(i == num_CGC_layers - 1)  # 最后一层只输出 task1/task2
            )
            for i in range(num_CGC_layers)
        ])

        # 两个任务各自的 Tower，输入维度应与 experts_out 保持一致
        self.tower1 = Tower(self.experts_out, 1, self.towers_hidden)
        self.tower2 = Tower(self.experts_out, 1, self.towers_hidden)

    def forward(self, x):
        """
        x: [batch_size, input_size] 原始输入特征

        返回：
        - final_output1: 任务1的预测（一般为 [batch_size, 1]）
        - final_output2: 任务2的预测
        """
        # 第一层 CGC：三个输入都用同一个原始特征 x
        emb = self.emb(x)
        cgc_outputs = self.cgc_layer1([emb, emb, emb])
        # cgc_outputs: [shared_out, task1_out, task2_out]
        # shape: list of 3 tensors, each [batch_size, experts_out]

        # 依次通过后续的 CGC 层
        for cgc_layer in self.cgc_layers:
            cgc_outputs = cgc_layer(cgc_outputs)
            # 中间层：仍然是 [shared_out, task1_out, task2_out]
            # 最后一层（if_last=True）：变为 [task1_out, task2_out]

        # 最后一层 CGC 的输出（任务级别表示）
        task1_repr = cgc_outputs[0]  # [batch_size, experts_out]
        task2_repr = cgc_outputs[1]  # [batch_size, experts_out]

        # 分别送入塔网络，得到最终预测
        final_output1 = self.tower1(task1_repr)  # [batch_size, 1]
        final_output2 = self.tower2(task2_repr)  # [batch_size, 1]

        return [final_output1, final_output2]
