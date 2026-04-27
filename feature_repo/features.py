"""
Feature view definitions for the fraud detection feature store.

These define WHAT features exist, WHERE they come from, and
HOW LONG they are valid.

Important:
    Velocity features (card1_count_1hr etc.) and aggregation features
    (card1_amt_mean, card1_amt_zscore etc.) are NOT defined here because
    they require real-time incremental updates that Feast's batch
    materialization cannot handle. Those are managed by
    src/feature_store/online_store.py using Redis sorted sets directly.

    This file covers all features that CAN be batch-materialized:
    - Static card features (encoded categoricals, C/D/V columns)
    - Time features (hour_of_day, is_night etc.)
    - Transaction-level features (TransactionAmt)
"""

from datetime import timedelta

from feast import FeatureService, FeatureView, Field
from feast.types import Float32, Float64, Int64

from feature_repo.data_sources import train_features_source
from feature_repo.entities import card_entity, email_entity

# Feature View 1: Card transaction features. Contains features associated with the card entity.
card_transaction_features = FeatureView(
    name="card_transaction_features",
    entities=[card_entity],
    ttl=timedelta(days=90),
    schema=[
        # Raw transaction features
        Field(name="TransactionAmt",        dtype=Float64),
        
        # Time-based features (stateless, computed from TransactionDT)
        Field(name="hour_of_day",           dtype=Int64),
        Field(name="day_of_week",           dtype=Int64),
        Field(name="days_since_start",      dtype=Float64),
        Field(name="is_night_transaction",  dtype=Int64),
        Field(name="is_weekend",            dtype=Int64),
        
        # Encoded categorical features
        Field(name="ProductCD_encoded",     dtype=Float64),
        Field(name="card4_encoded",         dtype=Float64),
        Field(name="card6_encoded",         dtype=Float64),
        Field(name="M1_encoded",            dtype=Float64),
        Field(name="M2_encoded",            dtype=Float64),
        Field(name="M3_encoded",            dtype=Float64),
        Field(name="M4_encoded",            dtype=Float64),
        Field(name="M5_encoded",            dtype=Float64),
        Field(name="M6_encoded",            dtype=Float64),
        
        # Raw C features (count-based, pass through as-is)
        Field(name="C1",  dtype=Float64),
        Field(name="C2",  dtype=Float64),
        Field(name="C3",  dtype=Float64),
        Field(name="C4",  dtype=Float64),
        Field(name="C5",  dtype=Float64),
        Field(name="C6",  dtype=Float64),
        Field(name="C7",  dtype=Float64),
        Field(name="C8",  dtype=Float64),
        Field(name="C9",  dtype=Float64),
        Field(name="C10", dtype=Float64),
        Field(name="C11", dtype=Float64),
        Field(name="C12", dtype=Float64),
        Field(name="C13", dtype=Float64),
        Field(name="C14", dtype=Float64),

        # D features (time delta)
        Field(name="D1",  dtype=Float64),
        Field(name="D2",  dtype=Float64),
        Field(name="D3",  dtype=Float64),
        Field(name="D4",  dtype=Float64),
        Field(name="D5",  dtype=Float64),
        Field(name="D10", dtype=Float64),
        Field(name="D11", dtype=Float64),
        Field(name="D15", dtype=Float64),
    ],
    online=True,
    source=train_features_source,
    tags={"team": "ml-engineering", "domain": "fraud"},
)


# Feature View 2: Email domain features
email_transaction_features = FeatureView(
    name="email_transaction_features",
    entities=[email_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="P_emaildomain_encoded", dtype=Float64),
        Field(name="R_emaildomain_encoded", dtype=Float64),
    ],
    online=True,
    source=train_features_source,
    tags={"team": "ml-engineering", "domain": "fraud"},
)

# Feature Service: groups features used by the fraud model. A FeatureService ties a specific set of features to a model version.
# When the model is retrained with different features, create a new FeatureService version.

fraud_model_v1 = FeatureService(
    name="fraud_model_v1",
    features=[
        card_transaction_features,
        email_transaction_features,
    ],
    description=(
        "Feature set for fraud detection model v1. "
        "Does not include velocity/aggregation features — "
        "those are served by the custom online store (Redis sorted sets)."
    ),
)
