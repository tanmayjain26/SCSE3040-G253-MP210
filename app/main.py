"""FastAPI prediction service for MP210 (Kochi Tea Growers' Estate).

POST /predict   -> {"prediction": <kg per worker>}
GET  /health    -> {"status": "ok"}
"""
import json
import logging
import sys
import time
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # make `tea` importable

from tea.clean import FEATURES, FLUSH_LEVELS, clean_flush  # noqa: E402

MODEL_PATH = ROOT / "models" / "model.joblib"
MODEL = None  # loaded on first use (tests can put a fake model here)


# ---------- JSON logging: one JSON line per prediction, to stdout ----------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


logger = logging.getLogger("mp210-api")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# ---------- request schema: the client's two rules live here ----------
class PredictRequest(BaseModel):
    worker_exp_yr: float = Field(..., ge=0, le=55)    # client rule 1: above 55 is impossible
    rain_mm_week: float = Field(..., ge=0)
    slope_deg: float = Field(..., ge=0)
    temp_c: float
    bush_age_yr: float = Field(..., ge=1.5)           # client rule 2: below 1.5 is a typing mistake
    hours_worked: float = Field(..., ge=0)
    flush: str
    section: Literal["Upper Garden", "Lower Garden", "River Side"]

    @field_validator("flush")
    @classmethod
    def flush_must_be_known(cls, value):
        cleaned = clean_flush(value)
        if cleaned not in FLUSH_LEVELS:
            raise ValueError(f"flush must be one of {FLUSH_LEVELS}")
        return cleaned


app = FastAPI(title="MP210 Tea Leaf Plucking")


def get_model():
    global MODEL
    if MODEL is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="model not available")
        MODEL = joblib.load(MODEL_PATH)
    return MODEL


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    model = get_model()
    row = pd.DataFrame([request.model_dump()])[FEATURES]
    started = time.time()
    prediction = float(model.predict(row)[0])
    # log the inputs and the answer: no names, no personal details
    logger.info("prediction", extra={"extra_fields": {
        "inputs": request.model_dump(),
        "prediction": round(prediction, 3),
        "latency_ms": round((time.time() - started) * 1000, 2),
    }})
    return {"prediction": prediction}
