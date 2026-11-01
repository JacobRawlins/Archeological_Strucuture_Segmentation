import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        dims = (0, 2, 3)

        intersection = (probs * targets).sum(dim=dims)

        denom = (
            probs.sum(dim=dims) +
            targets.sum(dim=dims)
        )

        dice = (
            2.0 * intersection + self.smooth
        ) / (denom + self.smooth)

        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return bce_loss + dice_loss