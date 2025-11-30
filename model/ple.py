import torch
import torch.nn as nn


class Expert(nn.Module):
    """
    单个 Expert：两层 MLP + ReLU + Dropout
    """
    def __init__(self, input_feature_size, expert_hidden_size, expert_output_size, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_feature_size, expert_hidden_size)
        self.fc2 = nn.Linear(expert_hidden_size, expert_output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, input_feature_size]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class CGC(nn.Module):
    """
    Customized Gate Control (CGC) 模块 —— 支持任意 num_tasks。

    每层：
    - 共享专家：num_shared_experts
    - 每个任务：num_specific_experts 个 task-specific 专家

    对于任务 t：
    - gate_t 在 [task_t 专属专家, 共享专家] 之间分配权重
    对于 shared：
    - gate_shared 在 [所有任务专属专家, 共享专家] 之间分配权重
    """
    def __init__(
        self,
        input_size: int,
        num_tasks: int,
        num_specific_experts: int,
        num_shared_experts: int,
        experts_out: int,
        experts_hidden: int,
        if_last: bool,
        expert_dropout: float = 0.3,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_tasks = num_tasks
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.if_last = if_last

        # 共享专家
        self.experts_shared = nn.ModuleList([
            Expert(self.input_size, self.experts_hidden, self.experts_out, dropout=expert_dropout)
            for _ in range(self.num_shared_experts)
        ])

        # 每个任务的专属专家：长度为 num_tasks 的 ModuleList
        self.experts_tasks = nn.ModuleList([
            nn.ModuleList([
                Expert(self.input_size, self.experts_hidden, self.experts_out, dropout=expert_dropout)
                for _ in range(self.num_specific_experts)
            ])
            for _ in range(self.num_tasks)
        ])

        # 每个任务自己的 gate：在 [task-specific, shared] 上分权重
        self.gate_tasks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_size, self.num_specific_experts + self.num_shared_experts),
                nn.Softmax(dim=1),
            )
            for _ in range(self.num_tasks)
        ])

        # shared gate：在 [所有任务的 task-specific, shared] 上分权重
        total_experts_for_shared = self.num_tasks * self.num_specific_experts + self.num_shared_experts
        self.gate_shared = nn.Sequential(
            nn.Linear(self.input_size, total_experts_for_shared),
            nn.Softmax(dim=1),
        )

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        inputs:
          非最后一层： [shared_input, task1_input, ..., taskN_input]
          最后一层：  同样形式

        返回：
          非最后一层： [shared_out, task1_out, ..., taskN_out]
          最后一层：   [task1_out, ..., taskN_out]
        """
        assert len(inputs) == self.num_tasks + 1, \
            f"CGC expects {self.num_tasks + 1} inputs (shared + {self.num_tasks} tasks), got {len(inputs)}"

        shared_input = inputs[0]
        task_inputs = inputs[1:]  # list 长度 = num_tasks
        batch_size = shared_input.size(0)

        # ====== 计算所有专家输出 ======
        # 共享专家输出：[num_shared_experts, batch, experts_out]
        experts_shared_out = torch.stack(
            [expert(shared_input) for expert in self.experts_shared],
            dim=0
        )

        # 各任务的专属专家输出：list，每个元素形状 [num_specific_experts, batch, experts_out]
        experts_tasks_out = []
        for t in range(self.num_tasks):
            task_out = torch.stack(
                [expert(task_inputs[t]) for expert in self.experts_tasks[t]],
                dim=0
            )
            experts_tasks_out.append(task_out)

        # ====== 每个任务自己的 gate 加权 ======
        task_outputs = []
        for t in range(self.num_tasks):
            gate_t = self.gate_tasks[t](task_inputs[t])  # [batch, num_specific_experts + num_shared_experts]

            # 拼接该任务的专属专家和共享专家
            experts_cat = torch.cat([experts_tasks_out[t], experts_shared_out], dim=0)
            # experts_cat: [num_specific_experts + num_shared_experts, batch, experts_out]

            # einsum: (a b c, b a) -> b c
            task_out = torch.einsum('abc,ba->bc', experts_cat, gate_t)
            task_outputs.append(task_out)  # [batch, experts_out]

        # ====== shared gate（如果不是最后一层） ======
        if not self.if_last:
            gate_shared = self.gate_shared(shared_input)  # [batch, total_experts_for_shared]

            # concat 所有任务的 task-specific 专家 + shared 专家
            experts_all = torch.cat(experts_tasks_out + [experts_shared_out], dim=0)
            # experts_all: [num_tasks * num_specific_experts + num_shared_experts, batch, experts_out]

            shared_out = torch.einsum('abc,ba->bc', experts_all, gate_shared)
            return [shared_out] + task_outputs
        else:
            return task_outputs  # 只返回各任务输出


class PLE(nn.Module):
    """
    PLE（Progressive Layered Extraction）：
    - 使用 user_feature_dict / item_feature_dict 管理特征
    - cate 特征建 Embedding，dense 特征直接拼接
    - hidden -> 多层 CGC -> 各任务 Tower -> logits
    """
    def __init__(
        self,
        user_feature_dict: dict,
        item_feature_dict: dict,
        task_name: str | None = None,
        emb_dim: int = 128,
        num_specific_experts: int = 1,
        num_shared_experts: int = 1,
        num_levels: int = 2,
        experts_hidden: int = 128,
        experts_out: int = 128,
        hidden_dim: list[int] = [128, 128],
        dropouts: list[float] = [0.5,0.5],
        output_size: int = 1,
        num_task: int = 2,
    ):
        super().__init__()

        self.task_name = task_name
        self.num_task = num_task

        if self.task_name == 'Tenrec':
            if user_feature_dict is None or item_feature_dict is None:
                raise Exception("user_feature_dict and item_feature_dict must be not None for Tenrec")
            if not isinstance(user_feature_dict, dict) or not isinstance(item_feature_dict, dict):
                raise Exception("user_feature_dict and item_feature_dict must be dict")
            
        elif self.task_name == 'census_income':
            if user_feature_dict is None:
                raise Exception("user_feature_dict must be not None for census_income")
            if not isinstance(user_feature_dict, dict):
                raise Exception("user_feature_dict must be dict")
            # census_income 只用 user_feature_dict
            item_feature_dict = {}

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict

        # embedding
        user_cate_feature_nums, item_cate_feature_nums = 0, 0

        for user_cate, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_cate_feature_nums += 1
                setattr(self, user_cate, nn.Embedding(num[0], emb_dim))

        for item_cate, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_cate_feature_nums += 1
                setattr(self, item_cate, nn.Embedding(num[0], emb_dim))

        hidden_size = (
            emb_dim * (user_cate_feature_nums + item_cate_feature_nums)
            + (len(self.user_feature_dict) - user_cate_feature_nums)
            + (len(self.item_feature_dict) - item_cate_feature_nums)
        )
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim

        # ------------------ 多层 CGC（PLE 核心） ------------------
        self.num_levels = num_levels
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden

        self.cgc_layers = nn.ModuleList()
        for level in range(num_levels):
            if level == 0:
                input_size = hidden_size   # 第一层输入 hidden
            else:
                input_size = experts_out   # 后续层输入 experts_out
            if_last = (level == num_levels - 1)

            self.cgc_layers.append(
                CGC(
                    input_size=input_size,
                    num_tasks=self.num_task,
                    num_specific_experts=self.num_specific_experts,
                    num_shared_experts=self.num_shared_experts,
                    experts_out=self.experts_out,
                    experts_hidden=self.experts_hidden,
                    if_last=if_last,
                )
            )

        #tower
        for i in range(self.num_task):
            setattr(self, f"task_{i + 1}_dnn", nn.ModuleList())
            tower_hid = [experts_out] + list(hidden_dim)
            for j in range(len(tower_hid) - 1):
                getattr(self, f"task_{i + 1}_dnn").add_module(
                    f"ctr_hidden_{j}", nn.Linear(tower_hid[j], tower_hid[j + 1])
                )
                getattr(self, f"task_{i + 1}_dnn").add_module(
                    f"ctr_batchnorm_{j}", nn.BatchNorm1d(tower_hid[j + 1])
                )
                getattr(self, f"task_{i + 1}_dnn").add_module(
                    f"ctr_dropout_{j}", nn.Dropout(dropouts[j])
                )
            getattr(self, f"task_{i + 1}_dnn").add_module(
                "task_last_layer", nn.Linear(tower_hid[-1], output_size)
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        x: [batch_size, num_features]
           列顺序必须和 user_feature_dict / item_feature_dict 一致。

        返回：
          task_outputs: 长度为 num_task 的 list，
                        每个元素 shape = [batch_size, output_size]（logits）
        """
        assert x.size(1) == len(self.user_feature_dict) + len(self.item_feature_dict)

        # ------------------ Embedding & hidden 融合 ------------------
        user_embed_list = []
        item_embed_list = []

        for user_feature, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_embed_list.append(getattr(self, user_feature)(x[:, num[1]].long()))
            else:
                user_embed_list.append(x[:, num[1]].unsqueeze(1))

        for item_feature, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_embed_list.append(getattr(self, item_feature)(x[:, num[1]].long()))
            else:
                item_embed_list.append(x[:, num[1]].unsqueeze(1))

        user_embed = torch.cat(user_embed_list, dim=1)
        if len(item_embed_list) > 0:
            item_embed = torch.cat(item_embed_list, dim=1)
            hidden = torch.cat([user_embed, item_embed], dim=1).float()
        else:
            hidden = user_embed.float()

        # ------------------ 通过多层 CGC ------------------
        # 第一层的输入：shared + 每个任务的输入（初始都用 hidden）
        cgc_input = [hidden] + [hidden for _ in range(self.num_task)]
        task_reprs = None

        for level, cgc_layer in enumerate(self.cgc_layers):
            cgc_output = cgc_layer(cgc_input)
            if level == self.num_levels - 1:
                # 最后一层：只返回 [task1_repr, ..., taskN_repr]
                task_reprs = cgc_output
            else:
                # 中间层：返回 [shared, task1, ..., taskN]
                cgc_input = cgc_output

        # ------------------ 每个任务通过自己的塔 ------------------
        task_outputs: list[torch.Tensor] = []
        for i in range(self.num_task):
            x_task = task_reprs[i]  # [batch, experts_out]
            for mod in getattr(self, f"task_{i + 1}_dnn"):
                x_task = mod(x_task)
            task_outputs.append(x_task)

        return task_outputs
