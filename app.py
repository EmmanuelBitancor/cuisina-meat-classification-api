from __future__ import annotations

import io
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

APP_TITLE = "Meat Classification (TFLite API)"

# Path to the .tflite file produced by convert_to_tflite.py
TFLITE_MODEL_PATH = Path(os.getenv("TFLITE_MODEL_PATH", "meat_model.tflite")).resolve()
LABELS_PATH = Path(os.getenv("LABELS_PATH", "labels.txt")).resolve()

# Optional: comma-separated labels for output indices (length should match output size).
# Example: "beef,chicken,lamb,pork,fish,goat,other"
LABELS_FROM_ENV = [
    s.strip()
    for s in os.getenv("LABELS", "").split(",")
    if s.strip()
]


def _load_labels_from_file(path: Path) -> list[str]:
    if not path.exists():
        return []

    labels: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        labels.append(line)
    return labels


# Priority: LABELS env var > labels.txt file.
LABELS = LABELS_FROM_ENV or _load_labels_from_file(LABELS_PATH)

app = FastAPI(title=APP_TITLE)

# Allow frontend apps (React/Vue/etc.) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_interpreter(model_path: Path) -> tf.lite.Interpreter:
    if not model_path.exists():
        raise FileNotFoundError(
            f"TFLite model not found at {model_path}. Run convert_to_tflite.py first."
        )
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


INTERPRETER = _load_interpreter(TFLITE_MODEL_PATH)
INPUT_DETAILS = INTERPRETER.get_input_details()
OUTPUT_DETAILS = INTERPRETER.get_output_details()
INFER_LOCK = threading.Lock()


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _looks_like_probabilities(x: np.ndarray, tol: float = 1e-3) -> bool:
    if x.ndim != 1 or x.size == 0:
        return False
    if not np.all(np.isfinite(x)):
        return False
    if np.any(x < -tol) or np.any(x > 1.0 + tol):
        return False
    total = float(np.sum(x))
    return abs(total - 1.0) <= 1e-2


def _preprocess_image(contents: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    img = img.convert("RGB")

    # Determine expected HxW from the TFLite input tensor.
    # Typical shape: [1, H, W, 3]
    input_shape = INPUT_DETAILS[0]["shape"]
    if len(input_shape) != 4:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected model input shape: {input_shape}",
        )

    height = int(input_shape[1])
    width = int(input_shape[2])

    img = img.resize((width, height))
    arr = np.asarray(img)

    # Match the model's input dtype.
    input_dtype = INPUT_DETAILS[0]["dtype"]
    if input_dtype == np.float32:
        x = arr.astype(np.float32)
    elif input_dtype == np.uint8:
        x = arr.astype(np.uint8)
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Unsupported input dtype: {input_dtype}",
        )

    # Add batch dimension.
    x = np.expand_dims(x, axis=0)
    return x


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_path": str(TFLITE_MODEL_PATH),
        "labels_path": str(LABELS_PATH),
        "labels_loaded": len(LABELS),
        "input": {
            "shape": INPUT_DETAILS[0]["shape"].tolist()
            if hasattr(INPUT_DETAILS[0]["shape"], "tolist")
            else list(INPUT_DETAILS[0]["shape"]),
            "dtype": str(INPUT_DETAILS[0]["dtype"]),
        },
        "output": {
            "shape": OUTPUT_DETAILS[0]["shape"].tolist()
            if hasattr(OUTPUT_DETAILS[0]["shape"], "tolist")
            else list(OUTPUT_DETAILS[0]["shape"]),
            "dtype": str(OUTPUT_DETAILS[0]["dtype"]),
        },
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=50),
) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    contents = await file.read()
    x = _preprocess_image(contents)

    # Run inference.
    with INFER_LOCK:
        INTERPRETER.set_tensor(INPUT_DETAILS[0]["index"], x)
        INTERPRETER.invoke()
        y = INTERPRETER.get_tensor(OUTPUT_DETAILS[0]["index"])  # [1, num_classes]

    scores = np.asarray(y).reshape(-1)

    # Dequantize if needed.
    out_quant = OUTPUT_DETAILS[0].get("quantization")
    if out_quant and isinstance(out_quant, tuple) and len(out_quant) == 2:
        scale, zero_point = out_quant
        if scale not in (0, 0.0):
            scores = (scores.astype(np.float32) - float(zero_point)) * float(scale)

    # Apply softmax only for logits. If the model already outputs probabilities,
    # re-applying softmax can flatten confidence values.
    scores = scores.astype(np.float32)
    if _looks_like_probabilities(scores):
        probs = scores
    else:
        probs = _softmax(scores)

    order = np.argsort(-probs)
    top = order[:top_k]

    preds: list[dict[str, Any]] = []
    for idx in top:
        label = LABELS[int(idx)] if len(LABELS) > int(idx) else None
        preds.append(
            {
                "index": int(idx),
                "label": label,
                "score": float(probs[int(idx)]),
            }
        )

    return {
        "filename": file.filename,
        "predictions": preds,
        "num_classes": int(probs.shape[0]),
    }
