import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.training.dataset import ShipDataset
from src.training.trainer import ShipSegmentationModule


class TestShipDataset:
    def test_getitem_shapes(self, tmp_path):
        img_dir = tmp_path / "train_v2"
        img_dir.mkdir()
        Image.new("RGB", (768, 768), (128, 128, 128)).save(img_dir / "img.jpg")

        df = pd.DataFrame({"ImageId": ["img.jpg"], "EncodedPixels": [["1 10"]]})
        sample = ShipDataset(img_dir, df, image_size=512)[0]

        assert sample["pixel_values"].shape == (3, 512, 512)
        assert sample["mask"].shape == (1, 512, 512)
        assert sample["mask"].sum() > 0


class TestShipSegmentationModule:
    def test_forward_shape(self):
        model = ShipSegmentationModule(
            model_name="nvidia/segformer-b0-finetuned-ade-512-512", lr=1e-4,
        )
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(2, 3, 512, 512))
        assert logits.shape == (2, 1, 128, 128)

    def test_training_step_returns_scalar_loss(self):
        model = ShipSegmentationModule(
            model_name="nvidia/segformer-b0-finetuned-ade-512-512", lr=1e-4,
        )
        batch = {
            "pixel_values": torch.randn(2, 3, 512, 512),
            "mask": torch.ones(2, 1, 512, 512),
        }
        loss = model._shared_step(batch, "train")
        assert loss.dim() == 0 and loss.item() > 0 and loss.requires_grad
