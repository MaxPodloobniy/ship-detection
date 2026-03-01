from pathlib import Path

import lightning
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.utils import rle_decode

# ImageNet statistics used by SegFormer pretrained weights
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class ShipDataset(Dataset):
    """Kaggle Airbus Ship Detection dataset.

    Expects a *pre-grouped* DataFrame where each row has:
    - ``ImageId``: filename inside *image_dir*
    - ``EncodedPixels``: **list** of RLE strings (may contain NaN entries)

    Args:
        image_dir: Directory with JPEG images.
        df: Pre-grouped DataFrame (one row per image).
        image_size: Spatial size to resize images and masks to.
    """

    def __init__(
        self,
        image_dir: Path,
        df: pd.DataFrame,
        image_size: int = 512,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.df = df.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_id = row["ImageId"]
        rle_list = row["EncodedPixels"]

        # ── image ─────────────────────────────────────────────────────
        img = Image.open(self.image_dir / image_id).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - IMAGENET_MEAN) / IMAGENET_STD

        # ── mask ──────────────────────────────────────────────────────
        mask = np.zeros((768, 768), dtype=np.uint8)
        for rle in rle_list:
            mask = np.maximum(mask, rle_decode(rle, shape=(768, 768)))

        mask_pil = Image.fromarray(mask)
        mask_pil = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask_pil)).unsqueeze(0).float()

        return {"pixel_values": img_tensor, "mask": mask_tensor}


class ShipDataModule(lightning.LightningDataModule):
    """Lightning DataModule for the Airbus Ship Detection dataset.

    Reads ``train_ship_segmentations_v2.csv``, groups RLE masks per image,
    and splits into train / val sets.

    Args:
        data_dir: Root directory containing ``train_v2/`` and the CSV.
        batch_size: Batch size for both train and val loaders.
        num_workers: DataLoader worker count.
        image_size: Spatial size passed to :class:`ShipDataset`.
        val_split: Fraction of data reserved for validation.
        seed: Random seed for the train/val split.
    """

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 16,
        num_workers: int = 4,
        image_size: int = 512,
        val_split: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.val_split = val_split
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        csv_path = self.data_dir / "train_ship_segmentations_v2.csv"
        df = pd.read_csv(csv_path)

        grouped = (
            df.groupby("ImageId")["EncodedPixels"]
            .apply(list)
            .reset_index()
        )

        train_df, val_df = train_test_split(
            grouped,
            test_size=self.val_split,
            random_state=self.seed,
        )

        image_dir = self.data_dir / "train_v2"
        self.train_dataset = ShipDataset(image_dir, train_df, self.image_size)
        self.val_dataset = ShipDataset(image_dir, val_df, self.image_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
