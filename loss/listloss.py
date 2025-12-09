import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
#from loss.BaseIntloss import BaseIntloss

#class Listloss(BaseIntloss):
class Listloss():
    def __init__(self,args):
        # 原本继承 BaseIntloss，这里简化为占位初始化
        self.cal_diversity = getattr(args, "cal_diversity", False)
        self.diversity_alpha = getattr(args, "diversity_alpha", 0.0)
    
    def list_loss(self,loss_matrix,valid_mask,rankings):
        """
        计算列表（Listwise）排序损失
        loss_matrix: [B, L, L]，模型的成对差值 s_i - s_j
        valid_mask : [B, L, L]，哪些 (i,j) 是合法 pair（都在 session_len 内并有有效排名）
        rankings   : [B, L]，每个 item 的排名（1 是最相关）

        思路：
        - 对每个 pair (i,j) 计算 exp(-(s_i - s_j))，差值越大惩罚越小
        - 只对 valid_mask=1 部分求和
        - 最后根据 ranking>0（排名有效）的位置求均值
        """

        # exp(-(score_i - score_j)) * mask
        loss_all = torch.exp(-loss_matrix) * valid_mask    # [B, L, L]

        # 对每一行（对应固定 item i）汇总 +1 避免 log(0)
        # rankings>0 表示该 item 是有效的候选
        loss_list = ((loss_all.sum(dim=2) + 1) * (rankings > 0)) \
                        .clamp(1).log().sum(dim=1) \
                        / (rankings > 0).sum(dim=-1)

        return loss_list.mean()
    

    def diversity(self,diff_matrix,base_diff,weights,valid_mask,rankings):
        """
        计算多样性正则项（Ambiguity Penalty）
        diff_matrix: [B, L, L]    集成模型的 s_i - s_j
        base_diff : [B, L, L, M]  每个基模型的 s_i - s_j
        weights   : [B, L, M]     用户对各基模型的权重
        valid_mask: [B, L, L]     合法位置
        rankings  : [B, L]        排名标签

        目标：
        - 让最终模型的 pairwise 差值更接近基模型差值的加权融合
        - 增强 ensemble 的“多样性”贡献
        """

        # exp(-(s_i - s_j))
        diff_exp = torch.exp(-diff_matrix)    # [B, L, L]

        # A_nk_up: 公式中的分子项 (平方项)
        # diff_exp.unsqueeze(3): [B, L, L, 1]
        # (base_diff - diff_matrix.unsqueeze(3)): 衡量 ensemble 与基模型差值的偏移量
        # sum(dim=2): 按 j 累积
        A_nk_up = (
            diff_exp.unsqueeze(3)
            * (base_diff - diff_matrix.unsqueeze(3))
            * valid_mask.unsqueeze(3)
        ).sum(dim=2) ** 2                      # [B, L, M]

        # 用 weights 对基模型贡献加权
        A_nk_w = (weights * A_nk_up).sum(-1)   # [B, L]

        # A_nk_bo: 分母项，防止过大
        A_nk_bo = 2 * (1 + (diff_exp * valid_mask).sum(dim=2)) ** 2    # [B, L]

        # 只对 rankings>0 的 item 求均值
        diversity_loss = (A_nk_w / A_nk_bo * (rankings > 0)).sum(dim=-1) \
            / (rankings > 0).sum(dim=-1)

        # 多样性 loss 是最大化的，因此取负号
        return -diversity_loss.mean()


    def forward(self,out_dict,in_batch):
        """
        计算最终损失：
            总损失 = list_loss + diversity_alpha * diversity_loss（可选）
        """

        ens_scores = out_dict['ens_score']     # [B, L] ensemble 最终分数
        device = ens_scores.device
        batch_size = in_batch['batch_size']

        # valid[i,j] = 两个位置都在 session_len 范围内
        valid = (torch.arange(ens_scores.size(1)).to(device)[None, :] #[1, L]
                 < in_batch['session_len'][:, None])     #[B, 1]        
		#[B,L]
        # 扩展为 [B, L, L] 的 pairwise mask
        valid_mask = valid.unsqueeze(2) * valid.unsqueeze(2).transpose(1, 2)

        # rankings: 未知标记置为 0
        rankings = torch.clamp(in_batch['ranking'], 0, 
                               max=in_batch['ranking'].max())     # [B, L]

        # ensemble 差值矩阵 s_i - s_j
        # [B, L, 1] - [B, 1, L]
        ens_diff = ens_scores.unsqueeze(2) - ens_scores.unsqueeze(2).transpose(1, 2)

        # diff_mask：只有 ranking 更高的 pair 才有效
        diff_mask = (rankings.unsqueeze(2) > rankings.unsqueeze(2).transpose(1, 2)) \
                    * valid_mask

        # Listwise 损失
        loss = self.list_loss(ens_diff, diff_mask, rankings)

        # 如果启用多样性正则项
        if self.cal_diversity:
            base_scores = in_batch['scores']         # [B, L, M]
            base_diff = base_scores.unsqueeze(2) - base_scores.unsqueeze(2).transpose(1, 2)   # [B, L, L, M]

            diversity_loss = self.diversity(
                ens_diff, base_diff,
                out_dict["weights"],  # ensemble 权重
                diff_mask, rankings
            )

            # diversity_alpha 是比例系数
            loss += diversity_loss * self.diversity_alpha

        # 该框架要求返回 3 个 loss（给不同阶段用）
        return loss
