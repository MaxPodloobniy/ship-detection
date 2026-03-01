import lightning
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from src.training.losses import BCEDiceLoss


class ShipSegmentationModule(lightning.LightningModule):
    """PyTorch Lightning module for binary ship segmentation with SegFormer.

    Loads a pretrained SegFormer encoder and replaces the decode head
    for single-class (ship / no-ship) output.

    Args:
        model_name: HuggingFace model identifier for SegFormer.
        lr: Learning rate for AdamW.
        bce_weight: Weight for the BCE component of the loss.
        dice_weight: Weight for the Dice component of the loss.
        pos_weight: Optional positive-class weight for BCE (class imbalance).
    """

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b1-finetuned-ade-512-512",
        lr: float = 1e-4,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=1,
            ignore_mismatched_sizes=True,
        )

        self.criterion = BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            pos_weight=pos_weight,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(B, 1, H/4, W/4)``."""
        return self.model(pixel_values=pixel_values).logits

    def _shared_step(
        self, batch: dict[str, torch.Tensor], stage: str
    ) -> torch.Tensor:
        pixel_values = batch["pixel_values"]
        masks = batch["mask"]

        logits = self.forward(pixel_values)

        # SegFormer outputs at 1/4 resolution — upsample to target size
        logits = F.interpolate(
            logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )

        loss = self.criterion(logits, masks)

        # IoU for monitoring
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            intersection = (preds * masks).sum()
            union = preds.sum() + masks.sum() - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=(stage == "train"))
        self.log(f"{stage}_iou", iou, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
