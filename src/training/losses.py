import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Operates on logits — applies sigmoid internally.

    Args:
        smooth: Smoothing constant to avoid division by zero and
                stabilise gradients when both prediction and target are empty.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Flatten spatial dims per sample: (B, -1)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1).float()

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy + Dice loss.

    Args:
        bce_weight: Multiplier for the BCE term.
        dice_weight: Multiplier for the Dice term.
        smooth: Smoothing constant forwarded to :class:`DiceLoss`.
        pos_weight: Optional positive-class weight forwarded to
                    ``F.binary_cross_entropy_with_logits`` to handle
                    class imbalance (ships occupy a small fraction of pixels).
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss(smooth=smooth)

        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.register_buffer("pos_weight", pw)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=self.pos_weight,
        )
        dice = self.dice(logits, targets)

        return self.bce_weight * bce + self.dice_weight * dice
