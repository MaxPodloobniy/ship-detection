from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from src.training.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.utils import rle_encode

TTA_TRANSFORMS = [
    ("orig", lambda x: x, lambda x: x),
    ("hflip", lambda x: torch.flip(x, dims=[-1]), lambda x: torch.flip(x, dims=[-1])),
    ("vflip", lambda x: torch.flip(x, dims=[-2]), lambda x: torch.flip(x, dims=[-2])),
    (
        "hvflip",
        lambda x: torch.flip(x, dims=[-2, -1]),
        lambda x: torch.flip(x, dims=[-2, -1]),
    ),
]


def mask_to_submission_rows(
    image_id: str, mask: np.ndarray, min_pixels: int = 10
) -> list[dict[str, str]]:
    """Split a binary mask into per-ship RLE rows using connected components."""
    num_labels, labels = cv2.connectedComponents(mask)
    rows: list[dict[str, str]] = []

    for label_id in range(1, num_labels):
        component = (labels == label_id).astype(np.uint8)
        if component.sum() < min_pixels:
            continue
        rle = rle_encode(component)
        if rle:
            rows.append({"ImageId": image_id, "EncodedPixels": rle})

    if not rows:
        rows.append({"ImageId": image_id, "EncodedPixels": np.nan})

    return rows


class InferenceImageDataset(Dataset):
    """Dataset that loads images from a directory for inference."""

    def __init__(self, image_dir: Path, image_size: int = 768) -> None:
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(
            p
            for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        self.transform = A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(
                    mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0
                ),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, str | torch.Tensor]:
        path = self.image_paths[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=img)
        return {"pixel_values": augmented["image"], "image_id": path.name}


class ShipPredictor:
    """Runs inference with a trained FPN ship-detection model.

    Args:
        checkpoint_path: Path to a Lightning .ckpt file.
        device: Torch device string ("cpu", "cuda", "mps").
        threshold: Probability threshold for binarising predicted masks.
        image_size: Spatial size the model was trained on.
        encoder_name: Encoder backbone used during training.
        use_tta: Enable test-time augmentation (4 flip variants).
        min_ship_pixels: Minimum connected component size to keep.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "cpu",
        threshold: float = 0.5,
        image_size: int = 768,
        encoder_name: str = "resnet34",
        use_tta: bool = False,
        min_ship_pixels: int = 10,
    ) -> None:
        self.device = torch.device(device)
        self.threshold = threshold
        self.image_size = image_size
        self.use_tta = use_tta
        self.min_ship_pixels = min_ship_pixels

        self.model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        state_dict = checkpoint["state_dict"]
        cleaned = {
            k.removeprefix("model."): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }
        self.model.load_state_dict(cleaned)
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def predict_batch(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass and return probabilities and binary masks.

        Args:
            pixel_values: (B, 3, H, W) normalised image tensor.

        Returns:
            probs: (B, 1, H, W) probability maps.
            masks: (B, 1, H, W) binary masks (uint8).
        """
        pixel_values = pixel_values.to(self.device)

        if self.use_tta:
            avg_probs = torch.zeros_like(self.model(pixel_values), dtype=torch.float32)
            for _name, forward_aug, reverse_aug in TTA_TRANSFORMS:
                augmented = forward_aug(pixel_values)
                logits = self.model(augmented)
                probs = torch.sigmoid(logits)
                avg_probs += reverse_aug(probs)
            avg_probs /= len(TTA_TRANSFORMS)
            probs = avg_probs
        else:
            logits = self.model(pixel_values)
            probs = torch.sigmoid(logits)

        masks = (probs > self.threshold).to(torch.uint8)
        return probs, masks

    def build_dataloader(
        self,
        image_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> DataLoader:
        dataset = InferenceImageDataset(image_dir, image_size=self.image_size)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type != "cpu",
        )

    def generate_submission(
        self, image_dir: Path, batch_size: int = 32, num_workers: int = 4
    ) -> pd.DataFrame:
        """Run inference on all images and produce a Kaggle submission DataFrame."""
        loader = self.build_dataloader(image_dir, batch_size, num_workers)
        rows: list[dict[str, str]] = []

        for batch in loader:
            _, masks = self.predict_batch(batch["pixel_values"])
            image_ids = batch["image_id"]

            for i, image_id in enumerate(image_ids):
                mask_np = masks[i, 0].cpu().numpy()
                mask_768 = cv2.resize(
                    mask_np,
                    (768, 768),
                    interpolation=cv2.INTER_NEAREST,
                )
                rows.extend(
                    mask_to_submission_rows(
                        image_id, mask_768, min_pixels=self.min_ship_pixels
                    )
                )

        return pd.DataFrame(rows, columns=["ImageId", "EncodedPixels"])

    def save_masks(
        self,
        image_dir: Path,
        output_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        """Run inference and save predicted masks as PNG images."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        loader = self.build_dataloader(image_dir, batch_size, num_workers)

        for batch in loader:
            _, masks = self.predict_batch(batch["pixel_values"])
            image_ids = batch["image_id"]

            for i, image_id in enumerate(image_ids):
                mask_np = masks[i, 0].cpu().numpy() * 255
                cv2.imwrite(
                    str(output_dir / f"{Path(image_id).stem}_mask.png"), mask_np
                )
