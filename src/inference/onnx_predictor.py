"""ONNX Runtime predictor for ship segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OnnxShipPredictor:
    """Run ship segmentation inference using an ONNX model.

    Args:
        model_path: Path to the ONNX model file.
        image_size: Spatial size the model expects.
        threshold: Probability threshold for binarising masks.
        min_ship_pixels: Minimum connected component size to keep.
    """

    def __init__(
        self,
        model_path: str | Path,
        image_size: int = 768,
        threshold: float = 0.5,
        min_ship_pixels: int = 10,
    ) -> None:
        self.image_size = image_size
        self.threshold = threshold
        self.min_ship_pixels = min_ship_pixels
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess an image for model input.

        Args:
            image: BGR or RGB image as (H, W, 3) uint8 array.

        Returns:
            (1, 3, H, W) float32 array, normalised with ImageNet stats.
        """
        if image.shape[2] == 4:
            image = image[:, :, :3]
        img = cv2.resize(image, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img.transpose(2, 0, 1)
        return img[np.newaxis]

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        """Run prediction on a single image.

        Args:
            image: BGR image as (H, W, 3) uint8 array (e.g. from cv2.imread).

        Returns:
            Dict with keys:
                - ship_count: number of detected ships.
                - mask: (H, W) uint8 binary mask (0 or 255).
                - has_ships: whether any ships were detected.
        """
        input_tensor = self.preprocess(image)
        logits = self.session.run(None, {"input": input_tensor})[0]

        probs = 1.0 / (1.0 + np.exp(-logits))
        binary = (probs > self.threshold).astype(np.uint8).squeeze()

        num_labels, labels = cv2.connectedComponents(binary)
        clean_mask = np.zeros_like(binary)
        ship_count = 0

        for label_id in range(1, num_labels):
            component = (labels == label_id).astype(np.uint8)
            if component.sum() >= self.min_ship_pixels:
                clean_mask |= component
                ship_count += 1

        return {
            "ship_count": ship_count,
            "mask": clean_mask * 255,
            "has_ships": ship_count > 0,
        }
