import cv2
import numpy as np
import pandas as pd
import torch

from src.training.dataset import ShipDataModule, ShipDataset
from src.training.trainer import ShipSegmentationModule


def _make_image(path, size=(768, 768), color=(128, 128, 128)):
    """Helper to create a test JPEG image."""
    img = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


class TestShipDataset:
    def test_getitem_shapes(self, tmp_path):
        img_dir = tmp_path / "train_v2"
        img_dir.mkdir()
        _make_image(img_dir / "img.jpg")

        df = pd.DataFrame({"ImageId": ["img.jpg"], "EncodedPixels": [["1 10"]]})
        sample = ShipDataset(img_dir, df)[0]

        assert sample["pixel_values"].shape == (3, 768, 768)
        assert sample["mask"].shape == (1, 768, 768)
        assert sample["mask"].sum() > 0

    def test_getitem_custom_image_size(self, tmp_path):
        img_dir = tmp_path / "train_v2"
        img_dir.mkdir()
        _make_image(img_dir / "img.jpg")

        df = pd.DataFrame({"ImageId": ["img.jpg"], "EncodedPixels": [["1 10"]]})
        sample = ShipDataset(img_dir, df, image_size=512)[0]

        assert sample["pixel_values"].shape == (3, 512, 512)
        assert sample["mask"].shape == (1, 512, 512)

    def test_train_augmentation_does_not_change_shape(self, tmp_path):
        img_dir = tmp_path / "train_v2"
        img_dir.mkdir()
        _make_image(img_dir / "img.jpg")

        df = pd.DataFrame({"ImageId": ["img.jpg"], "EncodedPixels": [["1 10"]]})
        ds = ShipDataset(img_dir, df, is_train=True)
        sample = ds[0]

        assert sample["pixel_values"].shape == (3, 768, 768)
        assert sample["mask"].shape == (1, 768, 768)

    def test_pixel_values_are_normalized(self, tmp_path):
        img_dir = tmp_path / "train_v2"
        img_dir.mkdir()
        _make_image(img_dir / "img.jpg", color=(255, 255, 255))

        df = pd.DataFrame({"ImageId": ["img.jpg"], "EncodedPixels": [[""]]})

        sample = ShipDataset(img_dir, df)[0]

        # After ImageNet normalization, pure white (255) should not be 1.0
        assert sample["pixel_values"].max() != 255.0
        assert sample["pixel_values"].dtype == torch.float32

    def test_empty_mask_is_all_zeros(self, tmp_path):
        img_dir = tmp_path / "train_v2"
        img_dir.mkdir()
        _make_image(img_dir / "img.jpg")

        df = pd.DataFrame({"ImageId": ["img.jpg"], "EncodedPixels": [[""]]})

        sample = ShipDataset(img_dir, df)[0]

        assert sample["mask"].sum() == 0


class TestShipSegmentationModule:
    def test_forward_shape(self):
        model = ShipSegmentationModule(encoder_name="resnet34", lr=1e-4)
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(2, 3, 256, 256))
        assert logits.shape == (2, 1, 256, 256)

    def test_training_step_returns_scalar_loss(self):
        model = ShipSegmentationModule(encoder_name="resnet34", lr=1e-4)
        batch = {
            "pixel_values": torch.randn(2, 3, 256, 256),
            "mask": torch.ones(2, 1, 256, 256),
        }
        loss = model._shared_step(batch, "train")
        assert loss.dim() == 0 and loss.item() > 0 and loss.requires_grad


class TestNegativeDownsampling:
    def test_downsampling_reduces_negatives(self, tmp_path):
        img_dir = tmp_path / "train_v2"
        img_dir.mkdir()
        csv_path = tmp_path / "train_ship_segmentations_v2.csv"

        rows = []
        for i in range(10):
            cv2.imwrite(
                str(img_dir / f"img{i}.jpg"),
                np.zeros((768, 768, 3), dtype=np.uint8),
            )
            if i < 2:
                rows.append({"ImageId": f"img{i}.jpg", "EncodedPixels": "1 10"})
            else:
                rows.append({"ImageId": f"img{i}.jpg", "EncodedPixels": np.nan})
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        dm = ShipDataModule(
            data_dir=tmp_path, negative_ratio=0.25, val_split=0.0001, seed=42
        )
        dm.setup()

        total = len(dm.train_dataset) + len(dm.val_dataset)
        assert total < 10
        assert total >= 3  # at least 2 positives + 1 negative
