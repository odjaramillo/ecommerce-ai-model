"""FastAPI application for e-commerce purchase intent prediction."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "ecommerce_pipeline.pkl"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Globals set during lifespan startup
model_pipeline = None
model_loaded = False


class PredictRequest(BaseModel):
    Administrative: int
    Administrative_Duration: float
    Informational: int
    Informational_Duration: float
    ProductRelated: int
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float
    Month: str
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: str
    Weekend: bool


class PredictResponse(BaseModel):
    classification: str
    probability: float
    human_readable_message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline, model_loaded
    logger.info(f"Loading model from {ARTIFACT_PATH} ...")
    if ARTIFACT_PATH.exists():
        model_pipeline = joblib.load(ARTIFACT_PATH)
        model_loaded = True
        logger.info("Model loaded successfully.")
    else:
        logger.error("Model artifact not found. Predictions will fail.")
        model_loaded = False
    yield
    logger.info("Shutting down API.")


app = FastAPI(title="E-commerce Purchase Intent API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/api/predict_intent", response_model=PredictResponse)
def predict_intent(request: PredictRequest):
    if not model_loaded or model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable.",
        )

    # Build DataFrame preserving column order expected by the pipeline
    data = {
        "Administrative": [request.Administrative],
        "Administrative_Duration": [request.Administrative_Duration],
        "Informational": [request.Informational],
        "Informational_Duration": [request.Informational_Duration],
        "ProductRelated": [request.ProductRelated],
        "ProductRelated_Duration": [request.ProductRelated_Duration],
        "BounceRates": [request.BounceRates],
        "ExitRates": [request.ExitRates],
        "PageValues": [request.PageValues],
        "SpecialDay": [request.SpecialDay],
        "Month": [request.Month],
        "OperatingSystems": [request.OperatingSystems],
        "Browser": [request.Browser],
        "Region": [request.Region],
        "TrafficType": [request.TrafficType],
        "VisitorType": [request.VisitorType],
        "Weekend": [request.Weekend],
    }
    df = pd.DataFrame(data)

    proba = model_pipeline.predict_proba(df)[0][1]
    return _build_response(proba)


def _build_response(proba: float) -> dict:
    """Build the standard JSON response from a positive-class probability."""
    probability = round(float(proba), 4)
    classification = "compra" if probability >= 0.5 else "no_compra"

    prob_pct = f"{probability * 100:.2f}"
    if probability >= 0.70:
        human_readable_message = (
            f"El usuario presenta un {prob_pct}% de probabilidades de hacer la compra, "
            f"lo que lo hace bastante probable"
        )
    elif probability >= 0.50:
        human_readable_message = (
            f"El usuario presenta un {prob_pct}% de probabilidades de hacer la compra, "
            f"lo que lo hace moderadamente probable"
        )
    else:
        human_readable_message = (
            f"El usuario presenta un {prob_pct}% de probabilidades de hacer la compra, "
            f"lo que lo hace poco probable"
        )

    return {
        "classification": classification,
        "probability": probability,
        "human_readable_message": human_readable_message,
    }


@app.post("/api/predict_intent_fast", response_model=PredictResponse)
def predict_intent_fast(request: PredictRequest):
    if not model_loaded or model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable.",
        )

    # Build NumPy array directly (shape 1x17), preserving column order
    array = np.array(
        [
            [
                request.Administrative,
                request.Administrative_Duration,
                request.Informational,
                request.Informational_Duration,
                request.ProductRelated,
                request.ProductRelated_Duration,
                request.BounceRates,
                request.ExitRates,
                request.PageValues,
                request.SpecialDay,
                request.Month,
                request.OperatingSystems,
                request.Browser,
                request.Region,
                request.TrafficType,
                request.VisitorType,
                request.Weekend,
            ]
        ],
        dtype=object,
    )

    proba = model_pipeline.predict_proba(array)[0][1]
    return _build_response(proba)
