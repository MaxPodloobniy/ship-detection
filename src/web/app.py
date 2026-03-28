"""Flask web application for ship detection."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from src.inference.onnx_predictor import OnnxShipPredictor

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app(
    model_path: str | Path | None = None,
    image_size: int = 768,
) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

    if model_path is None:
        model_path = os.environ.get("MODEL_PATH", "model.onnx")

    predictor = OnnxShipPredictor(model_path=model_path, image_size=image_size)

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/health")
    def health() -> tuple[dict, int]:
        return {"status": "ok"}, 200

    @app.route("/predict", methods=["POST"])
    def predict() -> tuple[dict, int]:
        if "file" not in request.files:
            return {"error": "No file provided"}, 400

        file = request.files["file"]
        if not file.filename or not _allowed_file(file.filename):
            return {"error": "Invalid file type. Use JPG or PNG."}, 400

        file_bytes = file.read()
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Could not decode image"}, 400

        result = predictor.predict(image)

        overlay = _create_overlay(image, result["mask"])
        _, buffer = cv2.imencode(".png", overlay)
        overlay_b64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify(
            {
                "ship_count": result["ship_count"],
                "has_ships": result["has_ships"],
                "overlay": overlay_b64,
            }
        )

    return app


def _create_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw a semi-transparent red overlay on detected ship regions."""
    overlay = cv2.resize(image, (mask.shape[1], mask.shape[0]))
    red = np.zeros_like(overlay)
    red[:, :, 2] = mask
    return cv2.addWeighted(overlay, 1.0, red, 0.4, 0)


def get_app() -> Flask:
    """Module-level app instance for gunicorn: ``gunicorn 'src.web.app:get_app()'``."""
    return create_app()


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=True)
