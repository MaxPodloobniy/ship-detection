import lightning
import segmentation_models_pytorch as smp
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from src.training.losses import BCEDiceLoss, BCELovaszLoss


class ShipSegmentationModule(lightning.LightningModule):
    """PyTorch Lightning module for binary ship segmentation with FPN.

    Args:
        encoder_name: Encoder backbone name (timm-compatible).
        encoder_weights: Pretrained weights identifier or None.
        lr: Learning rate for AdamW.
        bce_weight: Weight for the BCE component of the loss.
        dice_weight: Weight for the Dice component of the loss.
        lovasz_weight: Weight for the Lovász component of the loss.
        pos_weight: Optional positive-class weight for BCE (class imbalance).
        loss_type: Loss function to use ("bce_dice" or "bce_lovasz").
        scheduler_type: LR scheduler ("plateau" or "cosine").
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        lr: float = 1e-4,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        lovasz_weight: float = 0.5,
        pos_weight: float | None = None,
        loss_type: str = "bce_dice",
        scheduler_type: str = "plateau",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )

        self.criterion: BCELovaszLoss | BCEDiceLoss
        if loss_type == "bce_lovasz":
            self.criterion = BCELovaszLoss(
                bce_weight=bce_weight,
                lovasz_weight=lovasz_weight,
                pos_weight=pos_weight,
            )
        else:
            self.criterion = BCEDiceLoss(
                bce_weight=bce_weight,
                dice_weight=dice_weight,
                pos_weight=pos_weight,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (B, 1, H, W)."""
        return self.model(x)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        pixel_values = batch["pixel_values"]
        masks = batch["mask"]

        logits = self.forward(pixel_values)

        loss = self.criterion(logits, masks)

        # IoU for monitoring
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            intersection = (preds * masks).sum()
            union = preds.sum() + masks.sum() - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=(stage == "train"),
        )
        self.log(f"{stage}_iou", iou, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams["lr"], weight_decay=0.01
        )

        scheduler_type = self.hparams.get("scheduler_type", "plateau")

        scheduler: torch.optim.lr_scheduler.LRScheduler
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs or 10
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }
