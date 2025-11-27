import numpy as np
import torch
import torch.nn as nn

from Models_PLE import Expert, Tower


class CGC_OPE(nn.Module):
    """
    CGC_OPE: OPE 版本的 CGC（Conditional Gating Component）
    与原 PLE 的 CGC 不同之处在于：
      - 共享专家接收“全特征 embedding”
      - 任务专属专家接收“各自任务的优选特征 embedding”
    """
    def __init__(self, 
                 input_size_full,   # 共享专家看到的输入维度（第一层为 emb_dim_full，之后为 experts_out）
                 input_size_task1,  # 任务1 专属专家看到的输入维度
                 input_size_task2,  # 任务2 专属专家看到的输入维度
                 num_specific_experts, 
                 num_shared_experts, 
                 experts_out,       # 每个专家输出维度（也是下一层/塔网络的输入维度）
                 experts_hidden,    # 每个专家内部隐藏层维度
                 if_last):
        """
        input_size_full:   embedding 后的“全特征通道”维度
        input_size_task1:  embedding 后的“任务1 优选特征通道”维度
        input_size_task2:  embedding 后的“任务2 优选特征通道”维度
        if_last:           是否是最后一层 CGC_OPE（决定返回 [3 个输出] 还是 [2 个输出]）
        """
        super(CGC_OPE, self).__init__()
        self.if_last = if_last
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_out = experts_out
        
        # ========================
        # 1. 定义三组专家网络
        # ========================
        self.experts_shared = nn.ModuleList([
            Expert(input_size_full,  experts_hidden, experts_out) 
            for _ in range(num_shared_experts)
        ])  # 共享专家：只看“全特征通道”（如 x_full 的 embedding）

        self.experts_task1 = nn.ModuleList([
            Expert(input_size_task1, experts_hidden, experts_out) 
            for _ in range(num_specific_experts)
        ])  # 任务1 专属专家：只看“任务1 优选特征通道”（如 x_task1 的 embedding）

        self.experts_task2 = nn.ModuleList([
            Expert(input_size_task2, experts_hidden, experts_out) 
            for _ in range(num_specific_experts)
        ])  # 任务2 专属专家：只看“任务2 优选特征通道”
        
        # ========================
        # 2. 定义门控网络（Gate）
        # ========================
        # 共享门控：根据“全特征通道”决定下一层的 shared 输入
        # 输出长度 = 所有专家数量 = 2 * num_specific_experts + num_shared_experts
        self.gate_shared = nn.Sequential(
            nn.Linear(input_size_full, num_specific_experts * 2 + num_shared_experts),
            nn.Softmax(dim=1)
        )

        # 任务1 门控：根据“任务1 通道”选择（任务1 专家 + 共享专家）
        self.gate_task1 = nn.Sequential(
            nn.Linear(input_size_task1, num_specific_experts + num_shared_experts),
            nn.Softmax(dim=1)
        )

        # 任务2 门控：根据“任务2 通道”选择（任务2 专家 + 共享专家）
        self.gate_task2 = nn.Sequential(
            nn.Linear(input_size_task2, num_specific_experts + num_shared_experts),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x_full, x_task1, x_task2):
        """
        x_full:   (batch, input_size_full)   - 全特征 embedding / shared 通道输入
        x_task1:  (batch, input_size_task1)  - 任务1 优选特征 embedding / task1 通道输入
        x_task2:  (batch, input_size_task2)  - 任务2 优选特征 embedding / task2 通道输入

        第一层：
          - x_full/x_task1/x_task2 来自三个不同的 embedding
        后续层：
          - 三个分别来自上一层输出 [shared_out, task1_out, task2_out]
          - 三路维度相同，语义上仍可视作“shared/task1/task2 通道”
        """
        batch_size = x_full.size(0)
        
        # ================
        # 1. 先算所有专家输出
        # ================
        experts_shared_o = torch.stack(
            [e(x_full) for e in self.experts_shared],
            dim=0
        )  # (num_shared_experts, batch, experts_out)

        experts_task1_o = torch.stack(
            [e(x_task1) for e in self.experts_task1],
            dim=0
        )  # (num_specific_experts, batch, experts_out)

        experts_task2_o = torch.stack(
            [e(x_task2) for e in self.experts_task2],
            dim=0
        )  # (num_specific_experts, batch, experts_out)
        
        # ================
        # 2. 任务1 门控加权
        # ================
        selected_task1 = self.gate_task1(x_task1)  # (batch, num_specific_experts+num_shared_experts)

        gate_expert_output1 = torch.cat(
            (experts_task1_o, experts_shared_o),
            dim=0
        )  # (num_specific_experts+num_shared_experts, batch, experts_out)

        gate_task1_out = torch.einsum(
            'abc, ba -> bc',
            gate_expert_output1,   # [a, b, c]
            selected_task1         # [b, a]
        )  # -> (batch, experts_out)
        
        # ================
        # 3. 任务2 门控加权（同理）
        # ================
        selected_task2 = self.gate_task2(x_task2)  # (batch, num_specific_experts+num_shared_experts)
        gate_expert_output2 = torch.cat(
            (experts_task2_o, experts_shared_o),
            dim=0
        )  # (num_specific_experts+num_shared_experts, batch, experts_out)

        gate_task2_out = torch.einsum(
            'abc, ba -> bc',
            gate_expert_output2,
            selected_task2
        )  # (batch, experts_out)
        
        # ================
        # 4. 共享门控：为下一层生成 shared 通道输入
        # ================
        selected_shared = self.gate_shared(x_full)  # (batch, 2*num_specific_experts + num_shared_experts)

        gate_expert_output_shared = torch.cat(
            (experts_task1_o, experts_task2_o, experts_shared_o),
            dim=0
        )  # (2*num_specific_experts + num_shared_experts, batch, experts_out)

        gate_shared_out = torch.einsum(
            'abc, ba -> bc',
            gate_expert_output_shared,
            selected_shared
        )  # (batch, experts_out)
        
        # ================
        # 5. 输出
        # ================
        if self.if_last:
            return [gate_task1_out, gate_task2_out]
        else:
            return [gate_shared_out, gate_task1_out, gate_task2_out]


class OPE(nn.Module):
    """
    OPE: Optimized Private Expert
    核心思想：
      - 全特征走共享通道（shared experts）
      - 每个任务有自己的“优选特征”通道（private experts）
      - 通过 CGC_OPE（门控 + 专家）进行多层渐进抽取，最后送入各自任务塔
    """
    def __init__(self, 
                 input_size_full,   # 原始全特征维度（如所有特征拼在一起后的维度）
                 input_size_task1,  # 任务1 优选特征数量
                 input_size_task2,  # 任务2 优选特征数量
                 emb_dim_full,      # 全特征 embedding 后维度
                 emb_dim_task1,     # 任务1 优选特征 embedding 后维度
                 emb_dim_task2,     # 任务2 优选特征 embedding 后维度
                 num_CGC_layers,    # 额外堆叠的 CGC_OPE 层数（总层数 = 1 + num_CGC_layers，假设 >=1）
                 num_specific_experts, 
                 num_shared_experts,
                 experts_out, 
                 experts_hidden, 
                 towers_hidden):
        super(OPE, self).__init__()
        
        # 1. 三路独立的线性 embedding
        self.emb_full = nn.Linear(input_size_full,  emb_dim_full)
        self.emb_task1 = nn.Linear(input_size_task1, emb_dim_task1)
        self.emb_task2 = nn.Linear(input_size_task2, emb_dim_task2)
        
        # 2. 第一层 CGC_OPE
        self.cgc_ope_layer1 = CGC_OPE(
            input_size_full   = emb_dim_full,
            input_size_task1  = emb_dim_task1,
            input_size_task2  = emb_dim_task2,
            num_specific_experts = num_specific_experts,
            num_shared_experts   = num_shared_experts,
            experts_out      = experts_out,
            experts_hidden   = experts_hidden,
            if_last          = False
        )

        # 3. 后续 CGC_OPE 层
        self.cgc_ope_layers = nn.ModuleList([
            CGC_OPE(
                input_size_full   = experts_out,
                input_size_task1  = experts_out,
                input_size_task2  = experts_out,
                num_specific_experts = num_specific_experts,
                num_shared_experts   = num_shared_experts,
                experts_out      = experts_out,
                experts_hidden   = experts_hidden,
                if_last          = (i == num_CGC_layers - 1)
            )
            for i in range(num_CGC_layers)
        ])
        
        # 4. 任务塔
        self.tower1 = Tower(experts_out, 1, towers_hidden)
        self.tower2 = Tower(experts_out, 1, towers_hidden)
        
    def forward(self, x, x1, x2):
        """
        输入：
         - x:  原始全特征输入, shape (batch, input_size_full)
         - x1: 任务1优选特征输入, shape (batch, input_size_task1)
         - x2: 任务2优选特征输入, shape (batch, input_size_task2)
        """
        # 1. 三路分别做 embedding
        emb_full  = self.emb_full(x)   # (batch, emb_dim_full)
        emb_task1 = self.emb_task1(x1) # (batch, emb_dim_task1)
        emb_task2 = self.emb_task2(x2) # (batch, emb_dim_task2)
        
        # 2. 第一层 CGC_OPE
        cgc_out = self.cgc_ope_layer1(emb_full, emb_task1, emb_task2)
        
        # 3. 后续 CGC_OPE 层
        for layer in self.cgc_ope_layers:
            cgc_out = layer(cgc_out[0], cgc_out[1], cgc_out[2])
        
        # 4. 最后一层输出任务表示
        task1_repr = cgc_out[0]  # (batch, experts_out)
        task2_repr = cgc_out[1]  # (batch, experts_out)
        
        # 5. 送入各自塔网络
        out1 = self.tower1(task1_repr)  # (batch, 1)
        out2 = self.tower2(task2_repr)  # (batch, 1)
        
        return [out1, out2]


# ================================
# 新增：稀疏 gate 注意力（FeatureGate）
# ================================
class FeatureGate(nn.Module):
    """
    对每个任务的 embedding 维度做稀疏 gate：
      gate = sigmoid(MLP(x))
      out  = x * gate

    说明：
      - 这里不强制和为 1，而是每一维独立 in (0,1)
      - 你可以在 loss 中对 gate 加 L1/熵正则，鼓励稀疏：
          reg = lambda_ * (gate_task1.abs().mean() + gate_task2.abs().mean())
      - 最近一次 forward 的 gate 会保存在 self.last_gate 中，方便外部访问
    """
    def __init__(self, d_model, hidden_ratio: float = 1.0):
        super().__init__()
        hidden_dim = int(d_model * hidden_ratio)
        hidden_dim = max(hidden_dim, 1)

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.last_gate = None

    def forward(self, x):
        # x: (batch, d_model)
        logits = self.net(x)          # (batch, d_model)
        gate = torch.sigmoid(logits)  # (batch, d_model), in (0,1)
        self.last_gate = gate
        return x * gate               # (batch, d_model)


class HFSOPE(nn.Module):
    """
    HFSOPE（Attention-augmented OPE，稀疏 gate 版本）

    改动点：
      1）保留 OPE 的任务私有子空间：
          - 输入仍然是 x_full / x1 / x2 三路
          - x1 / x2 来自 PLE/特征重要性筛选出的优选特征
      2）在任务私有通道 embedding 后，引入 FeatureGate 做细粒度 gate：
          emb_task1' = FeatureGate(emb_task1)
          emb_task2' = FeatureGate(emb_task2)

    shared 通道 emb_full 不做 gate，保持原始 OPE 的稳健性。
    """
    def __init__(self, 
                 input_size_full,
                 input_size_task1,
                 input_size_task2,
                 emb_dim_full,
                 emb_dim_task1,
                 emb_dim_task2,
                 num_CGC_layers,
                 num_specific_experts,
                 num_shared_experts,
                 experts_out,
                 experts_hidden,
                 towers_hidden,
                 gate_hidden_ratio: float = 1.0):
        super(HFSOPE, self).__init__()

        # 1. 三路 embedding（与 OPE 完全一致）
        self.emb_full  = nn.Linear(input_size_full,  emb_dim_full)
        self.emb_task1 = nn.Linear(input_size_task1, emb_dim_task1)
        self.emb_task2 = nn.Linear(input_size_task2, emb_dim_task2)

        # 2. 任务私有通道上的稀疏 gate 注意力
        self.gate_task1 = FeatureGate(emb_dim_task1, hidden_ratio=gate_hidden_ratio)
        self.gate_task2 = FeatureGate(emb_dim_task2, hidden_ratio=gate_hidden_ratio)

        # 3. 第一层 CGC_OPE（输入仍然是三路通道，只是任务通道被 gate 过）
        self.cgc_ope_layer1 = CGC_OPE(
            input_size_full      = emb_dim_full,
            input_size_task1     = emb_dim_task1,
            input_size_task2     = emb_dim_task2,
            num_specific_experts = num_specific_experts,
            num_shared_experts   = num_shared_experts,
            experts_out          = experts_out,
            experts_hidden       = experts_hidden,
            if_last              = False
        )

        # 4. 后续 CGC_OPE 层（与 OPE 相同）
        self.cgc_ope_layers = nn.ModuleList([
            CGC_OPE(
                input_size_full      = experts_out,
                input_size_task1     = experts_out,
                input_size_task2     = experts_out,
                num_specific_experts = num_specific_experts,
                num_shared_experts   = num_shared_experts,
                experts_out          = experts_out,
                experts_hidden       = experts_hidden,
                if_last              = (i == num_CGC_layers - 1)
            )
            for i in range(num_CGC_layers)
        ])

        # 5. 任务塔（与 OPE 相同）
        self.tower1 = Tower(experts_out, 1, towers_hidden)
        self.tower2 = Tower(experts_out, 1, towers_hidden)

    def forward(self, x, x1, x2):
        """
        输入：
          - x  : 全特征
          - x1 : 任务1 优选特征
          - x2 : 任务2 优选特征
        """
        # 1. 三路 embedding
        emb_full  = self.emb_full(x)    # shared 通道
        emb_task1 = self.emb_task1(x1)  # 任务1 私有通道
        emb_task2 = self.emb_task2(x2)  # 任务2 私有通道

        # 2. 在任务私有子空间上做稀疏 gate 注意力（细粒度调节）
        emb_task1 = self.gate_task1(emb_task1)
        emb_task2 = self.gate_task2(emb_task2)

        # 3. 第一层 CGC_OPE
        cgc_out = self.cgc_ope_layer1(emb_full, emb_task1, emb_task2)

        # 4. 后续 CGC_OPE 层
        for layer in self.cgc_ope_layers:
            cgc_out = layer(cgc_out[0], cgc_out[1], cgc_out[2])

        # 5. 最后一层输出任务表示
        task1_repr = cgc_out[0]
        task2_repr = cgc_out[1]

        # 6. 塔预测
        out1 = self.tower1(task1_repr)
        out2 = self.tower2(task2_repr)

        return [out1, out2]
