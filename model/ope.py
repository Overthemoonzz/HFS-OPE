import numpy as np
import torch
import torch.nn as nn

class Expert(nn.Module):
    """
    单个 Expert：一层 Linear + ReLU（无 Dropout）
    """
    def __init__(self, input_feature_size, expert_output_size):
        super().__init__()
        self.fc = nn.Linear(input_feature_size, expert_output_size)  # 对应 MMOE 里的 W_e, b_e
        self.relu = nn.ReLU()  # 对齐 mmoe.py 里的 expert_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, input_feature_size]
        out = self.fc(x)       # [batch_size, expert_output_size]
        out = self.relu(out)
        return out

class CGC_OPE(nn.Module):
    """
    CGC_OPE: OPE 版本的 CGC（Conditional Gating Component）
    与原 PLE 的 CGC 不同之处在于：
      - 共享专家接收“全特征 embedding”
      - 任务专属专家接收“各自任务的优选特征 embedding”
    """
    def __init__(self, 
                 input_size_share,   # 共享专家看到的输入维度（第一层为 emb_dim_full，之后为 experts_out）
                 input_size_tasks,  # 任务专属专家看到的输入维度, 一个列表
                 num_specific_experts, 
                 num_shared_experts, 
                 experts_out,       # 每个专家输出维度（也是下一层/塔网络的输入维度）
                 experts_hidden,    # 每个专家内部隐藏层维度
                 if_last):
        """
        input_size_share:   embedding 后的“全特征通道”维度
        if_last:           是否是最后一层 CGC_OPE（决定返回 [3 个输出] 还是 [2 个输出]）
        """
        super(CGC_OPE, self).__init__()
        self.if_last = if_last
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_out = experts_out
        self.num_tasks = len(input_size_tasks)
        # ========================
        # 1. 定义共享专家和任务专家
        # ========================
        self.experts_shared = nn.ModuleList([
            Expert(input_size_share, experts_out) 
            for _ in range(num_shared_experts)
        ])  # 共享专家：只看“全特征通道”（如 x_full 的 embedding）

        for i in range(len(input_size_tasks)):
            setattr(self, f"experts_task_{i}", nn.ModuleList([
                Expert(input_size_tasks[i], experts_out)
                for _ in range(num_specific_experts)
            ])) 
        
        # ========================
        # 2. 定义门控网络（Gate）
        # ========================
        # 共享门控：根据“全特征通道”决定下一层的 shared 输入
        # 输出长度 = 所有专家数量 = num_expert * num_specific_experts + num_shared_experts
        self.gate_shared = nn.Sequential(
            nn.Linear(input_size_share, num_specific_experts * len(input_size_tasks) + num_shared_experts),
            nn.Softmax(dim=1)
        )

        # 任务门控：根据“任务通道”选择（任务专家 + 共享专家）

        self.gate_tasks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size_tasks[i], num_specific_experts + num_shared_experts),
                nn.Softmax(dim=1),
            )
            for i in range(len(input_size_tasks))
        ])

        
    def forward(self, x_full, x_tasks):
        """
        x_full:   (batch, input_size_share)   - 全特征 embedding / shared 通道输入
        x_tasks:  ((batch, input_task0), (batch, input_task1), ...)  - 任务i 优选特征 embedding / taski 通道输入

        """
        
        # ================
        # 1. 先算所有专家输出
        # ================
        experts_shared_o = torch.stack(
            [e(x_full) for e in self.experts_shared],
            dim=0
        )  # (num_shared_experts, batch, experts_out)

        experts_task_os = []
        for i in range(self.num_tasks):
            x_task = x_tasks[i]
            experts_task = getattr(self, f"experts_task_{i}")
            experts_task_o = torch.stack([e(x_task) for e in experts_task], dim=0)
            experts_task_os.append(experts_task_o)
        # ================
        # 2. 任务门控加权
        # ================

        gate_task_outs = []
        for i in range(self.num_tasks):
            x_task = x_tasks[i]
            selected_task = self.gate_tasks[i](x_task)
            gate_expert_output = torch.cat(
                (experts_task_os[i], experts_shared_o),
                dim=0
            )  # (num_specific_experts+num_shared_experts, batch, experts_out)
            gate_task_out = torch.einsum(
                'abc, ba -> bc',
                gate_expert_output,   # [a, b, c]
                selected_task         # [b, a]
            )  # -> (batch, experts_out)
            gate_task_outs.append(gate_task_out)
        
        # ================
        # 3. 共享门控：为下一层生成 shared 通道输入
        # ================
        selected_shared = self.gate_shared(x_full)  # (batch, num_tasks*num_specific_experts + num_shared_experts)
        experts_all = torch.cat(
            experts_task_os + [experts_shared_o],
            dim=0
        )  # (num_tasks*num_specific_experts + num_shared_experts, batch, experts_out)

        gate_shared_out = torch.einsum(
            'abc,ba->bc',
            experts_all,        # (a, b, c)
            selected_shared,    # (b, a)
        )
        
        # ================
        # 5. 输出
        # ================
        if self.if_last:
            return gate_task_outs
        else:
            return gate_shared_out, gate_task_outs

class OPE(nn.Module):
    """
    改动点：
      1) 任务私有子空间：
          - 输入是 x_full / x_1 / x_2
          - x1 / x2 / ... xi 来自 PLE/特征重要性筛选出的优选特征

    """
    def __init__(self, 
                 num_tasks,
                 user_feature_dict: dict,
                 item_feature_dict: dict,
                 context_feature_dict: dict,
                 task_feats, #dict
                 emb_dim = 128, #假设每一个头的embedding size 相同以简化代码
                 num_CGC_layers = 2,
                 num_specific_experts = 2,
                 num_shared_experts = 2,
                 experts_out = 128,
                 experts_hidden = 128,
                 hidden_dim = [128, 128],
                 dropouts = [0.2, 0.2],
                 output_size = 1):
        super(OPE, self).__init__()

        self.num_tasks = num_tasks
        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.context_feature_dict = context_feature_dict
        self.task_feats = task_feats
        # embedding
        user_cate_feature_nums, item_cate_feature_nums, context_cate_feature_nums = 0, 0, 0 # share
        task_user_cate_feature_nums, task_item_cate_feature_nums, task_context_cate_feature_nums = [0 for _ in range(num_tasks)], [0 for _ in range(num_tasks)], [0 for _ in range(num_tasks)]
        
        for user_cate, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_cate_feature_nums += 1
                for t in range(num_tasks):
                    if user_cate in task_feats[t]:
                        task_user_cate_feature_nums[t] += 1
                setattr(self, f"{user_cate}_embs", nn.ModuleList([
                    nn.Embedding(num[0], emb_dim) for _ in range(num_tasks + 1)
                ]))
                        
        for item_cate, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_cate_feature_nums += 1
                for t in range(num_tasks):
                    if item_cate in task_feats[t]:
                        task_item_cate_feature_nums[t] += 1
                setattr(self, f"{item_cate}_embs", nn.ModuleList([
                    nn.Embedding(num[0], emb_dim) for _ in range(num_tasks + 1)
                ]))

        for context_cate, num in self.context_feature_dict.items():
            if num[0] > 1:
                context_cate_feature_nums += 1
                for t in range(num_tasks):
                    if context_cate in task_feats[t]:
                        task_context_cate_feature_nums[t] += 1
                setattr(self, f"{context_cate}_embs", nn.ModuleList([
                    nn.Embedding(num[0], emb_dim) for _ in range(num_tasks + 1)
                ]))
                
                
        input_size_share = (
            emb_dim * (user_cate_feature_nums + context_cate_feature_nums + item_cate_feature_nums)
            + (len(self.user_feature_dict) - user_cate_feature_nums)
            + (len(self.item_feature_dict) - item_cate_feature_nums)
            + (len(self.context_feature_dict) - context_cate_feature_nums)
        )

        input_size_tasks = [0 for _ in range(num_tasks)]
        for i in range(num_tasks):
            input_size_tasks[i] = (
            emb_dim * (task_user_cate_feature_nums[i] + task_item_cate_feature_nums[i] + task_context_cate_feature_nums[i])
            + (len(task_feats[i]) - task_user_cate_feature_nums[i] - task_item_cate_feature_nums[i] - task_context_cate_feature_nums[i])
        )

        # 2. 第一层 CGC_OPE（输入仍然是三路通道，只是任务通道被 gate 过）
        self.cgc_ope_layer1 = CGC_OPE(
            input_size_share     = input_size_share,
            input_size_tasks     = input_size_tasks,
            num_specific_experts = num_specific_experts,
            num_shared_experts   = num_shared_experts,
            experts_out          = experts_out,
            experts_hidden       = experts_hidden,
            if_last              = False
        )

        # 3. 后续 CGC_OPE 层（与 OPE 相同）
        self.cgc_ope_layers = nn.ModuleList([
            CGC_OPE(
                input_size_share     = experts_out,
                input_size_tasks     = [experts_out] * num_tasks,
                num_specific_experts = num_specific_experts,
                num_shared_experts   = num_shared_experts,
                experts_out          = experts_out,
                experts_hidden       = experts_hidden,
                if_last              = (i == num_CGC_layers - 1)
            )
            for i in range(num_CGC_layers)
        ])

        # 4. 任务塔
        for i in range(self.num_tasks):
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

    def forward(self, x):
        """
        输入：
          - x  : 全特征
        """
        user_embed_list_share, user_embed_list_tasks= [],[[] for _ in range(self.num_tasks)]
        item_embed_list_share, item_embed_list_tasks= [],[[] for _ in range(self.num_tasks)]
        context_embed_list_share, context_embed_list_tasks= [],[[] for _ in range(self.num_tasks)]

        for user_feature, num in self.user_feature_dict.items():
            if num[0] > 1:
                emb_layers = getattr(self, f"{user_feature}_embs")
                share_emb = emb_layers[0](x[:, num[1]].long())
                user_embed_list_share.append(share_emb)
                for i in range(self.num_tasks):
                    if user_feature in self.task_feats[i]:
                        task_emb = emb_layers[i+1](x[:, num[1]].long())
                        user_embed_list_tasks[i].append(task_emb)
            else:
                user_embed_list_share.append(x[:, num[1]].unsqueeze(1))
                for i in range(self.num_tasks):
                    if user_feature in self.task_feats[i]:
                        user_embed_list_tasks[i].append(x[:, num[1]].unsqueeze(1))

        for item_feature, num in self.item_feature_dict.items():
            if num[0] > 1:
                emb_layers = getattr(self, f"{item_feature}_embs")
                share_emb = emb_layers[0](x[:, num[1]].long())
                item_embed_list_share.append(share_emb)
                for i in range(self.num_tasks):
                    if item_feature in self.task_feats[i]:
                        task_emb = emb_layers[i+1](x[:, num[1]].long())
                        item_embed_list_tasks[i].append(task_emb)
            else:
                item_embed_list_share.append(x[:, num[1]].unsqueeze(1))
                for i in range(self.num_tasks):
                    if item_feature in self.task_feats[i]:
                        item_embed_list_tasks[i].append(x[:, num[1]].unsqueeze(1))

        for context_feature, num in self.context_feature_dict.items():
            if num[0] > 1:
                emb_layers = getattr(self, f"{context_feature}_embs")
                share_emb = emb_layers[0](x[:, num[1]].long())
                context_embed_list_share.append(share_emb)
                for i in range(self.num_tasks):
                    if context_feature in self.task_feats[i]:
                        task_emb = emb_layers[i+1](x[:, num[1]].long())
                        context_embed_list_tasks[i].append(task_emb)
            else:
                context_embed_list_share.append(x[:, num[1]].unsqueeze(1))
                for i in range(self.num_tasks):
                    if context_feature in self.task_feats[i]:
                        context_embed_list_tasks[i].append(x[:, num[1]].unsqueeze(1))

        user_embed_share = torch.cat(user_embed_list_share, dim=1)
        item_embed_share = torch.cat(item_embed_list_share, dim=1)
        context_embed_share = torch.cat(context_embed_list_share, dim=1)
        share_hidden = torch.cat([user_embed_share, item_embed_share, context_embed_share], dim=1).float()

        task_hidden = [[] for _ in range(self.num_tasks)]
        for i in range(self.num_tasks):
            parts = []
            if user_embed_list_tasks[i]:
                user_embed_task = torch.cat(user_embed_list_tasks[i], dim=1)
                parts.append(user_embed_task)
            if item_embed_list_tasks[i]:
                item_embed_task = torch.cat(item_embed_list_tasks[i], dim=1)
                parts.append(item_embed_task)
            if context_embed_list_tasks[i]:
                context_embed_task = torch.cat(context_embed_list_tasks[i], dim=1)
                parts.append(context_embed_task)
            if len(parts) > 1:
                task_hidden[i] = torch.cat(parts, dim=1).float()
            elif len(parts) == 1:
                task_hidden[i] = parts[0].float() 

        # 2. 第一层 CGC_OPE
        shared, task_outs = self.cgc_ope_layer1(share_hidden, task_hidden)

        # 3. 后续 CGC_OPE 层
        for layer in self.cgc_ope_layers:
            out = layer(shared, task_outs)
            if layer.if_last:
                task_outs = out
            else:
                shared, task_outs = out

        # 4. 最后一层输出任务表示
        task_reprs = task_outs

        # 5. 塔预测
        out = [None] * self.num_tasks
        for i in range(self.num_tasks):
            x_task = task_reprs[i]                        # [batch, experts_out]
            dnn = getattr(self, f"task_{i + 1}_dnn")      # ModuleList
            for mod in dnn:
                x_task = mod(x_task)
            out[i] = x_task                               # [batch, output_size]
        return out
