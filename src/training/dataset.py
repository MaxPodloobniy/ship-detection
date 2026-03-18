from pathlib import Path

import albumentations as A
import cv2
import lightning
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.utils import rle_decode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ShipDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        df: pd.DataFrame,
        image_size: int = 768,
        is_train: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.df = df.reset_index(drop=True)
        self.image_size = image_size

        transforms_list = [A.Resize(height=self.image_size, width=self.image_size)]

        if is_train:
            transforms_list.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ])

        transforms_list.extend([
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
            ToTensorV2(),
        ])

        self.transform = A.Compose(transforms_list)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_id = row["ImageId"]
        rle_list = row["EncodedPixels"]

        img = cv2.imread(str(self.image_dir / image_id))
        assert img is not None, f"Failed to load image: {image_id}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = np.zeros((768, 768), dtype=np.uint8)
        for rle in rle_list:
            mask = np.maximum(mask, rle_decode(rle, shape=(768, 768)))

        augmented = self.transform(image=img, mask=mask)
        img_tensor = augmented["image"]
        mask_tensor = augmented["mask"].unsqueeze(0).float()

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
        negative_ratio: Fraction of ship-free images to keep (0.0–1.0).
        seed: Random seed for the train/val split and downsampling.
    """

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 16,
        num_workers: int = 4,
        image_size: int = 768,
        val_split: float = 0.1,
        negative_ratio: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.val_split = val_split
        self.negative_ratio = negative_ratio
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        csv_path = self.data_dir / "train_ship_segmentations_v2.csv"
        df = pd.read_csv(csv_path)

        grouped = (
            df.groupby("ImageId")["EncodedPixels"]
            .apply(list)
            .reset_index()
        )

        # ── negative-class downsampling ───────────────────────────────
        if self.negative_ratio < 1.0:
            has_ship = grouped["EncodedPixels"].apply(
                lambda rles: any(isinstance(r, str) for r in rles)
            )
            positives = grouped[has_ship]
            negatives = grouped[~has_ship].sample(
                frac=self.negative_ratio, random_state=self.seed
            )
            grouped = pd.concat([positives, negatives], ignore_index=True)

        train_df, val_df = train_test_split(
            grouped,
            test_size=self.val_split,
            random_state=self.seed,
        )

        image_dir = self.data_dir / "train_v2"
        self.train_dataset = ShipDataset(image_dir, train_df, self.image_size, is_train=True)
        self.val_dataset = ShipDataset(image_dir, val_df, self.image_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
