"""Optional local FastAPI wrapper for the same Transformers model."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Support Ticket Classifier")
model_path = Path(os.getenv("MODEL_PATH", "outputs/model"))
classifier = pipeline("text-classification", model=str(model_path), tokenizer=str(model_path))


class PredictionRequest(BaseModel):
    text: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, object]:
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    return {"prediction": classifier(request.text)[0]}
