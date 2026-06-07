import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.prediction_pipeline import PredictionPipeline

SATISFACTION_LABELS = [
    "Very Dissatisfied",
    "Dissatisfied",
    "Neutral",
    "Satisfied",
    "Very Satisfied",
]


@st.cache_data
def load_field_options():
    """Load categorical options from the training CSV."""
    csv_path = ROOT / "data/raw/eCommerce_Customer_support_data.csv"
    df = pd.read_csv(csv_path)

    return {
        "channel_name": sorted(df["channel_name"].dropna().unique()),
        "category": sorted(df["category"].dropna().unique()),
        "Sub-category": sorted(df["Sub-category"].dropna().unique()),
        "Agent_name": sorted(df["Agent_name"].dropna().unique()),
        "Supervisor": sorted(df["Supervisor"].dropna().unique()),
        "Manager": sorted(df["Manager"].dropna().unique()),
        "Tenure Bucket": sorted(df["Tenure Bucket"].dropna().unique()),
        "Agent Shift": sorted(df["Agent Shift"].dropna().unique()),
    }


@st.cache_resource
def load_prediction_pipeline():
    """Load trained model and preprocessor once."""
    pipeline = PredictionPipeline(
        paths_config=str(ROOT / "configs/paths.yaml")
    )
    pipeline.load_artifacts()
    return pipeline


def build_input_dataframe(form_data: dict) -> pd.DataFrame:
    """Build a single-row dataframe matching the raw CSV schema."""
    return pd.DataFrame([{
        "Unique id": str(uuid.uuid4()),
        "channel_name": form_data["channel_name"],
        "category": form_data["category"],
        "Sub-category": form_data["Sub-category"],
        "Customer Remarks": form_data.get("Customer Remarks") or None,
        "Order_id": form_data["Order_id"],
        "order_date_time": None,
        "Issue_reported at": form_data["Issue_reported at"],
        "issue_responded": form_data["issue_responded"],
        "Survey_response_Date": form_data["Survey_response_Date"],
        "Customer_City": None,
        "Product_category": None,
        "Item_price": None,
        "connected_handling_time": None,
        "Agent_name": form_data["Agent_name"],
        "Supervisor": form_data["Supervisor"],
        "Manager": form_data["Manager"],
        "Tenure Bucket": form_data["Tenure Bucket"],
        "Agent Shift": form_data["Agent Shift"],
    }])


def render_gauge(score: int):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={
            "reference": 3,
            "increasing": {"color": "green"},
            "decreasing": {"color": "red"},
        },
        title={
            "text": "Predicted CSAT Score",
            "font": {"size": 24, "color": "white"},
        },
        gauge={
            "axis": {
                "range": [1, 5],
                "tickwidth": 1,
                "tickcolor": "black",
                "tickvals": [1, 2, 3, 4, 5],
                "ticktext": ["😡", "🙁", "😐", "🙂", "😍"],
            },
            "bar": {"color": "royalblue", "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [1, 2], "color": "#ff4d4d"},
                {"range": [2, 3], "color": "#ffa64d"},
                {"range": [3, 4], "color": "#ffe066"},
                {"range": [4, 5], "color": "#28a745"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    st.plotly_chart(fig, use_container_width=True)


def render_probabilities(probabilities: np.ndarray):
    labels = [
        "😡 Very Dissatisfied",
        "🙁 Dissatisfied",
        "😐 Neutral",
        "🙂 Satisfied",
        "😍 Very Satisfied",
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=probabilities,
        y=labels,
        orientation="h",
        marker_color="mediumseagreen",
    ))
    fig.update_layout(
        title="Prediction Confidence (Softmax Probabilities)",
        xaxis_title="Confidence Score",
        yaxis_title="CSAT Level",
        xaxis=dict(range=[0, 1]),
        height=400,
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(
        page_title="CSAT Prediction",
        layout="wide",
    )
    st.title("Customer Satisfaction Prediction System")
    st.caption(
        "Powered by the trained model on "
        "eCommerce_Customer_support_data.csv"
    )

    try:
        options = load_field_options()
        pipeline = load_prediction_pipeline()
    except Exception as exc:
        st.error(
            "Could not load model artifacts. "
            "Run `python main.py train` first."
        )
        st.exception(exc)
        return

    input_col, result_col = st.columns([1, 2])

    with input_col:
        st.header("Input Details")
        with st.form("prediction_form"):
            channel_name = st.selectbox(
                "Channel Name", options["channel_name"]
            )
            category = st.selectbox(
                "Category", options["category"]
            )
            sub_category = st.selectbox(
                "Sub-category", options["Sub-category"]
            )
            agent_name = st.selectbox(
                "Agent Name", options["Agent_name"]
            )
            supervisor = st.selectbox(
                "Supervisor", options["Supervisor"]
            )
            manager = st.selectbox(
                "Manager", options["Manager"]
            )
            tenure_bucket = st.selectbox(
                "Tenure Bucket", options["Tenure Bucket"]
            )
            agent_shift = st.selectbox(
                "Agent Shift", options["Agent Shift"]
            )
            customer_remarks = st.text_area(
                "Customer Remarks (optional)",
                placeholder="Leave blank if none",
            )
            order_id = st.text_input(
                "Order ID",
                value=str(uuid.uuid4()),
            )
            issue_reported_at = st.text_input(
                "Issue Reported At",
                value="01/08/2023 11:13",
            )
            issue_responded = st.text_input(
                "Issue Responded At",
                value="01/08/2023 11:47",
            )
            survey_response_date = st.text_input(
                "Survey Response Date",
                value="01-Aug-23",
            )

            submit_button = st.form_submit_button(
                "Predict CSAT Score"
            )

    if submit_button:
        try:
            form_data = {
                "channel_name": channel_name,
                "category": category,
                "Sub-category": sub_category,
                "Agent_name": agent_name,
                "Supervisor": supervisor,
                "Manager": manager,
                "Tenure Bucket": tenure_bucket,
                "Agent Shift": agent_shift,
                "Customer Remarks": customer_remarks or None,
                "Order_id": order_id,
                "Issue_reported at": issue_reported_at,
                "issue_responded": issue_responded,
                "Survey_response_Date": survey_response_date,
            }

            input_df = build_input_dataframe(form_data)
            probabilities = pipeline.predict_proba(input_df)[0]
            predicted_score = int(np.argmax(probabilities) + 1)

            with result_col:
                st.header("Prediction Results")
                render_gauge(predicted_score)
                render_probabilities(probabilities)
                st.info(
                    "The customer is predicted to be: "
                    f"**{SATISFACTION_LABELS[predicted_score - 1]}** "
                    f"(CSAT {predicted_score})"
                )

        except Exception as exc:
            st.error(f"Error during prediction: {exc}")


if __name__ == "__main__":
    main()
