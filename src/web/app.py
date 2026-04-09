"""FastAPI web application for ship detection."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.inference.onnx_predictor import OnnxShipPredictor

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16 MB

_WEB_DIR = Path(__file__).parent


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app(
    model_path: str | Path | None = None,
    image_size: int = 768,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Ship Detection")

    app.mount(
        "/static",
        StaticFiles(directory=_WEB_DIR / "static"),
        name="static",
    )
    templates = Jinja2Templates(directory=_WEB_DIR / "templates")

    if model_path is None:
        model_path = os.environ.get("MODEL_PATH", "model.onnx")

    predictor = OnnxShipPredictor(model_path=model_path, image_size=image_size)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name="index.html")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict[str, object]:
        if not file.filename or not _allowed_file(file.filename):
            raise HTTPException(
                status_code=400, detail="Invalid file type. Use JPG or PNG."
            )

        file_bytes = await file.read()
        if len(file_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File too large")

        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        result = predictor.predict(image)

        overlay = _create_overlay(image, result["mask"])
        _, buffer = cv2.imencode(".png", overlay)
        overlay_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

        return {
            "ship_count": result["ship_count"],
            "has_ships": result["has_ships"],
            "overlay": overlay_b64,
        }

    return app


def _create_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw a semi-transparent red overlay on detected ship regions."""
    overlay = cv2.resize(image, (mask.shape[1], mask.shape[0]))
    red = np.zeros_like(overlay)
    red[:, :, 2] = mask
    return cv2.addWeighted(overlay, 1.0, red, 0.4, 0)


app: FastAPI | None = None
if os.environ.get("MODEL_PATH"):
    app = create_app()
