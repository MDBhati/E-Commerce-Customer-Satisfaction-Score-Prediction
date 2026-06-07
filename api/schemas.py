"""Pydantic schemas for the CSAT prediction API."""

from pydantic import BaseModel, Field


class CSATPredictionRequest(BaseModel):
    """Single customer support record for CSAT prediction."""

    channel_name: str = Field(
        ...,
        examples=["Inbound"],
        description="Communication channel",
    )
    category: str = Field(
        ...,
        examples=["Order Related"],
        description="Issue category",
    )
    sub_category: str = Field(
        ...,
        alias="Sub-category",
        examples=["Delayed"],
        description="Issue sub-category",
    )
    agent_name: str = Field(
        ...,
        alias="Agent_name",
        examples=["Richard Buchanan"],
    )
    supervisor: str = Field(
        ...,
        alias="Supervisor",
        examples=["Mason Gupta"],
    )
    manager: str = Field(
        ...,
        alias="Manager",
        examples=["Jennifer Nguyen"],
    )
    tenure_bucket: str = Field(
        ...,
        alias="Tenure Bucket",
        examples=["On Job Training"],
    )
    agent_shift: str = Field(
        ...,
        alias="Agent Shift",
        examples=["Morning"],
    )
    order_id: str = Field(
        ...,
        alias="Order_id",
        examples=["c27c9bb4-fa36-4140-9f1f-21009254ffdb"],
    )
    issue_reported_at: str = Field(
        ...,
        alias="Issue_reported at",
        examples=["01/08/2023 11:13"],
    )
    issue_responded: str = Field(
        ...,
        alias="issue_responded",
        examples=["01/08/2023 11:47"],
    )
    survey_response_date: str = Field(
        ...,
        alias="Survey_response_Date",
        examples=["01-Aug-23"],
    )
    customer_remarks: str | None = Field(
        default=None,
        alias="Customer Remarks",
        examples=["Very good"],
    )
    unique_id: str | None = Field(
        default=None,
        alias="Unique id",
    )

    model_config = {"populate_by_name": True}


class CSATPredictionResponse(BaseModel):
    """Prediction result for a single record."""

    csat_score: int = Field(..., ge=1, le=5)
    probabilities: dict[str, float]
    satisfaction_label: str


class BatchPredictionRequest(BaseModel):
    records: list[CSATPredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: list[CSATPredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
