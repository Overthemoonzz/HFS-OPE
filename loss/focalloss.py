import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with logits.
    Supports pos_weight (same meaning as BCEWithLogitsLoss).
    """
    def __init__(self, gamma=2.0, pos_weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight,
            reduction="none"
        )
    def forward(self, logits, targets):
        # 1. Per-sample BCE loss (no reduction)
        bce = self.bce(logits, targets)
        # 2. Compute probability p = sigmoid(logits)
        prob = torch.sigmoid(logits)

        # 3. Compute pt = p for y=1, and (1-p) for y=0
        pt = prob * targets + (1 - prob) * (1 - targets)

        # 4. focal factor
        focal_factor = (1 - pt) ** self.gamma

        # 5. Apply focal factor
        loss = focal_factor * bce  # shape [B]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss