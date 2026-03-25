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
        pw = self.pos_weight if isinstance(self.pos_weight, torch.Tensor) else None
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=pw,
        )
        dice = self.dice(logits, targets)

        return self.bce_weight * bce + self.dice_weight * dice


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of the Lovász extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary Lovász hinge loss, flat version."""
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class LovaszLoss(nn.Module):
    """Lovász-Hinge loss for binary segmentation, per-image then averaged."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = []
        for logit, target in zip(
            logits.view(logits.size(0), -1), targets.view(targets.size(0), -1)
        ):
            losses.append(lovasz_hinge_flat(logit, target))
        return torch.stack(losses).mean()


class BCELovaszLoss(nn.Module):
    """Combined Binary Cross-Entropy + Lovász loss.

    Args:
        bce_weight: Multiplier for the BCE term.
        lovasz_weight: Multiplier for the Lovász term.
        pos_weight: Optional positive-class weight for BCE.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        lovasz_weight: float = 0.5,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.lovasz = LovaszLoss()

        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.register_buffer("pos_weight", pw)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = self.pos_weight if isinstance(self.pos_weight, torch.Tensor) else None
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=pw,
        )
        lovasz = self.lovasz(logits, targets)

        return self.bce_weight * bce + self.lovasz_weight * lovasz
