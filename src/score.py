"""AML Managed Online Endpoint scoring script."""

import json
import os
from pathlib import Path

from transformers import pipeline


_classifier = None


def init() -> None:
    global _classifier
    model_dir = Path(os.environ["AZUREML_MODEL_DIR"])
    model_path = next((path for path in model_dir.rglob("config.json")), None)
    if model_path is None:
        raise FileNotFoundError(f"No model config found below {model_dir}")
    model_path = model_path.parent
    _classifier = pipeline("text-classification", model=str(model_path), tokenizer=str(model_path))


def run(raw_data: str | dict) -> dict[str, object]:
    payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    texts = payload.get("texts", [payload.get("text", "")])
    if not all(isinstance(text, str) and text for text in texts):
        raise ValueError("Request must contain a non-empty 'text' or 'texts' field")
    return {"predictions": _classifier(texts)}
