from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.training.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.training.trainer import ShipSegmentationModule
from src.utils import rle_encode


class InferenceImageDataset(Dataset):
    """Simple dataset that loads images from a directory for inference.

    Args:
        image_dir: Directory with JPEG images.
        image_size: Spatial size to resize images to.
    """

    def __init__(self, image_dir: Path, image_size: int = 768) -> None:
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(
            p
            for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, str | torch.Tensor]:
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = (
            img_tensor - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        ) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return {"pixel_values": img_tensor, "image_id": path.name}


class ShipPredictor:
    """Runs inference with a trained ShipSegmentationModule.

    Args:
        checkpoint_path: Path to a Lightning ``.ckpt`` file.
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        threshold: Probability threshold for binarising predicted masks.
        image_size: Spatial size the model was trained on.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "cpu",
        threshold: float = 0.5,
        image_size: int = 768,
    ) -> None:
        self.device = torch.device(device)
        self.threshold = threshold
        self.image_size = image_size

        self.model = ShipSegmentationModule.load_from_checkpoint(
            checkpoint_path, map_location=self.device
        )
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def predict_batch(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass and return full-res probabilities and binary masks.

        Args:
            pixel_values: ``(B, 3, H, W)`` normalised image tensor.

        Returns:
            probs: ``(B, 1, 768, 768)`` probability maps.
            masks: ``(B, 1, 768, 768)`` binary masks (uint8).
        """
        pixel_values = pixel_values.to(self.device)
        logits = self.model(pixel_values)

        # Upsample to original Kaggle resolution for RLE encoding
        logits = F.interpolate(
            logits, size=(768, 768), mode="bilinear", align_corners=False
        )

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
        self,
        image_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> pd.DataFrame:
        """Run inference on all images and produce a Kaggle submission DataFrame.

        Returns:
            DataFrame with columns ``ImageId`` and ``EncodedPixels``.
        """
        loader = self.build_dataloader(image_dir, batch_size, num_workers)
        rows: list[dict[str, str]] = []

        for batch in loader:
            _, masks = self.predict_batch(batch["pixel_values"])
            image_ids = batch["image_id"]

            for i, image_id in enumerate(image_ids):
                mask_np = masks[i, 0].cpu().numpy()
                rle = rle_encode(mask_np)
                rows.append(
                    {
                        "ImageId": image_id,
                        "EncodedPixels": rle if rle else "",
                    }
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
                Image.fromarray(mask_np).save(
                    output_dir / f"{Path(image_id).stem}_mask.png"
                )
