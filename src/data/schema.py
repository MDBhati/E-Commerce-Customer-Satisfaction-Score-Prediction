"""
Schema definition for eCommerce_Customer_support_data.csv.

Source: data/raw/eCommerce_Customer_support_data.csv
Shape: (85907, 20)
Target: CSAT Score (1-5)
"""

TARGET_COLUMN = "CSAT Score"
VALID_CSAT_SCORES = {1, 2, 3, 4, 5}

CSAT_SCHEMA = {
    "Unique id": "object",
    "channel_name": "object",
    "category": "object",
    "Sub-category": "object",
    "Customer Remarks": "object",
    "Order_id": "object",
    "order_date_time": "object",
    "Issue_reported at": "object",
    "issue_responded": "object",
    "Survey_response_Date": "object",
    "Customer_City": "object",
    "Product_category": "object",
    "Item_price": "float64",
    "connected_handling_time": "float64",
    "Agent_name": "object",
    "Supervisor": "object",
    "Manager": "object",
    "Tenure Bucket": "object",
    "Agent Shift": "object",
    "CSAT Score": "int64",
}

# Columns dropped during preprocessing (>70% missing)
HIGH_MISSING_COLUMNS = [
    "connected_handling_time",
    "Customer_City",
    "Product_category",
    "Item_price",
    "order_date_time",
]

# Categorical features used for one-hot encoding
CATEGORICAL_COLUMNS = [
    "channel_name",
    "category",
    "Sub-category",
    "Agent_name",
    "Supervisor",
    "Manager",
    "Tenure Bucket",
    "Agent Shift",
]

# Identifier / free-text columns excluded from model features
IDENTIFIER_COLUMNS = [
    "Unique id",
    "Order_id",
    "Customer Remarks",
]

DATETIME_COLUMNS = [
    "Issue_reported at",
    "issue_responded",
    "Survey_response_Date",
]
