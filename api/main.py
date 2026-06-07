"""
FastAPI inference service for CSAT score prediction.

Run:
    uvicorn api.main:app --reload --port 8000
"""

import sys
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CSATPredictionRequest,
    CSATPredictionResponse,
    HealthResponse,
)
from src.pipeline.prediction_pipeline import PredictionPipeline

SATISFACTION_LABELS = {
    1: "Very Dissatisfied",
    2: "Dissatisfied",
    3: "Neutral",
    4: "Satisfied",
    5: "Very Satisfied",
}

pipeline: PredictionPipeline | None = None


def _request_to_dataframe(
    record: CSATPredictionRequest,
) -> pd.DataFrame:
    """Convert API request to a single-row dataframe matching CSV schema."""

    data = {
        "Unique id": record.unique_id or str(uuid.uuid4()),
        "channel_name": record.channel_name,
        "category": record.category,
        "Sub-category": record.sub_category,
        "Customer Remarks": record.customer_remarks,
        "Order_id": record.order_id,
        "order_date_time": None,
        "Issue_reported at": record.issue_reported_at,
        "issue_responded": record.issue_responded,
        "Survey_response_Date": record.survey_response_date,
        "Customer_City": None,
        "Product_category": None,
        "Item_price": None,
        "connected_handling_time": None,
        "Agent_name": record.agent_name,
        "Supervisor": record.supervisor,
        "Manager": record.manager,
        "Tenure Bucket": record.tenure_bucket,
        "Agent Shift": record.agent_shift,
    }

    return pd.DataFrame([data])


def _build_response(
    score: int,
    probabilities: list[float],
) -> CSATPredictionResponse:
    prob_map = {
        str(i + 1): round(float(p), 4)
        for i, p in enumerate(probabilities)
    }

    return CSATPredictionResponse(
        csat_score=int(score),
        probabilities=prob_map,
        satisfaction_label=SATISFACTION_LABELS[int(score)],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""

    global pipeline

    try:
        pipeline = PredictionPipeline()
        pipeline.load_artifacts()
    except Exception as exc:
        pipeline = None
        print(
            f"Warning: model artifacts not loaded — {exc}",
            file=sys.stderr,
        )

    yield


app = FastAPI(
    title="E-Commerce CSAT Prediction API",
    description=(
        "Predict customer satisfaction (CSAT) scores from "
        "e-commerce support interaction data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok" if pipeline is not None else "degraded",
        model_loaded=pipeline is not None,
    )


@app.post("/predict", response_model=CSATPredictionResponse)
def predict(record: CSATPredictionRequest):
    """Predict CSAT score for a single support interaction."""

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Train the model first with "
                "`python main.py train`."
            ),
        )

    try:
        df = _request_to_dataframe(record)
        predictions = pipeline.predict(df)
        probabilities = pipeline.predict_proba(df)[0]

        return _build_response(
            score=int(predictions[0]),
            probabilities=probabilities.tolist(),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """Predict CSAT scores for multiple support interactions."""

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Train the model first with "
                "`python main.py train`."
            ),
        )

    if not request.records:
        raise HTTPException(
            status_code=400,
            detail="At least one record is required.",
        )

    try:
        frames = [
            _request_to_dataframe(record)
            for record in request.records
        ]
        df = pd.concat(frames, ignore_index=True)

        predictions = pipeline.predict(df)
        probabilities = pipeline.predict_proba(df)

        results = [
            _build_response(
                score=int(score),
                probabilities=proba.tolist(),
            )
            for score, proba in zip(predictions, probabilities)
        ]

        return BatchPredictionResponse(predictions=results)

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
