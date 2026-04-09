FROM python:3.11-slim

WORKDIR /app

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY src/__init__.py /app/src/__init__.py
COPY src/inference/__init__.py /app/src/inference/__init__.py
COPY src/inference/onnx_predictor.py /app/src/inference/onnx_predictor.py
COPY src/web/ /app/src/web/

ARG MODEL_PATH=model.onnx
COPY ${MODEL_PATH} /app/model.onnx

ENV MODEL_PATH=/app/model.onnx
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["uvicorn", "src.web.app:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]
