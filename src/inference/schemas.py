from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TransactionInput(BaseModel):
    TransactionID: int = Field(
        ...,
        description="Unique transaction identifier",
        json_schema_extra={"example": 3663592},
    )

    TransactionDT: int = Field(
        ...,
        description=(
            "Seconds elapsed since reference point (86400 = day 1). "
            "Used to compute time-based features."
        ),
        json_schema_extra={"example": 86400},
    )

    TransactionAmt: float = Field(
        ...,
        description="Transaction amount in USD",
        gt=0,
        json_schema_extra={"example": 68.50},
    )

    # --- Product and card fields ---
    ProductCD: Optional[str] = Field(
        None, description="Product code", json_schema_extra={"example": "W"}
    )
    card1: Optional[int] = Field(
        None, description="Card identifier 1", json_schema_extra={"example": 13926}
    )
    card2: Optional[float] = Field(None, json_schema_extra={"example": 358.0})
    card3: Optional[float] = Field(None, json_schema_extra={"example": 150.0})
    card4: Optional[str] = Field(
        None, description="Card network", json_schema_extra={"example": "visa"}
    )
    card5: Optional[float] = Field(None, json_schema_extra={"example": 226.0})
    card6: Optional[str] = Field(
        None, description="Card type", json_schema_extra={"example": "debit"}
    )

    # --- Address fields ---
    addr1: Optional[float] = Field(
        None, description="Billing address zip", json_schema_extra={"example": 315.0}
    )
    addr2: Optional[float] = Field(
        None, description="Billing address country", json_schema_extra={"example": 87.0}
    )

    # --- Distance fields ---
    dist1: Optional[float] = Field(None, json_schema_extra={"example": 19.0})
    dist2: Optional[float] = Field(None)

    # --- Email domains ---
    P_emaildomain: Optional[str] = Field(
        None,
        description="Purchaser email domain",
        json_schema_extra={"example": "gmail.com"},
    )
    R_emaildomain: Optional[str] = Field(
        None,
        description="Recipient email domain",
        json_schema_extra={"example": "gmail.com"},
    )

    # --- Count features C1-C14 ---
    C1: Optional[float] = Field(None)
    C2: Optional[float] = Field(None)
    C3: Optional[float] = Field(None)
    C4: Optional[float] = Field(None)
    C5: Optional[float] = Field(None)
    C6: Optional[float] = Field(None)
    C7: Optional[float] = Field(None)
    C8: Optional[float] = Field(None)
    C9: Optional[float] = Field(None)
    C10: Optional[float] = Field(None)
    C11: Optional[float] = Field(None)
    C12: Optional[float] = Field(None)
    C13: Optional[float] = Field(None)
    C14: Optional[float] = Field(None)

    # --- Time delta features D1-D15 ---
    D1: Optional[float] = Field(None)
    D2: Optional[float] = Field(None)
    D3: Optional[float] = Field(None)
    D4: Optional[float] = Field(None)
    D5: Optional[float] = Field(None)
    D6: Optional[float] = Field(None)
    D7: Optional[float] = Field(None)
    D8: Optional[float] = Field(None)
    D9: Optional[float] = Field(None)
    D10: Optional[float] = Field(None)
    D11: Optional[float] = Field(None)
    D12: Optional[float] = Field(None)
    D13: Optional[float] = Field(None)
    D14: Optional[float] = Field(None)
    D15: Optional[float] = Field(None)

    # --- M features (match flags) M1-M9 ---
    M1: Optional[str] = Field(None)
    M2: Optional[str] = Field(None)
    M3: Optional[str] = Field(None)
    M4: Optional[str] = Field(None)
    M5: Optional[str] = Field(None)
    M6: Optional[str] = Field(None)
    M7: Optional[str] = Field(None)
    M8: Optional[str] = Field(None)
    M9: Optional[str] = Field(None)

    # --- Identity features id_01 to id_38 ---
    # Numeric identity features
    id_01: Optional[float] = Field(None)
    id_02: Optional[float] = Field(None)
    id_03: Optional[float] = Field(None)
    id_04: Optional[float] = Field(None)
    id_05: Optional[float] = Field(None)
    id_06: Optional[float] = Field(None)
    id_07: Optional[float] = Field(None)
    id_08: Optional[float] = Field(None)
    id_09: Optional[float] = Field(None)
    id_10: Optional[float] = Field(None)
    id_11: Optional[float] = Field(None)
    id_13: Optional[float] = Field(None)
    id_14: Optional[float] = Field(None)
    id_17: Optional[float] = Field(None)
    id_19: Optional[float] = Field(None)
    id_20: Optional[float] = Field(None)
    id_32: Optional[float] = Field(None)

    id_12: Optional[str] = Field(None)
    id_15: Optional[str] = Field(None)
    id_16: Optional[str] = Field(None)
    id_28: Optional[str] = Field(None)
    id_29: Optional[str] = Field(None)
    id_30: Optional[str] = Field(None)
    id_31: Optional[str] = Field(None)
    id_33: Optional[str] = Field(None)
    id_34: Optional[str] = Field(None)
    id_35: Optional[str] = Field(None)
    id_36: Optional[str] = Field(None)
    id_37: Optional[str] = Field(None)
    id_38: Optional[str] = Field(None)
    DeviceType: Optional[str] = Field(None)
    DeviceInfo: Optional[str] = Field(None)

    # --- V features V1-V339 ---
    # There are 339 V features. Rather than listing all of them (which would make this file enormous), we accept them via model_extra.
    model_config = {"extra": "allow"}

    @field_validator("TransactionAmt")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"TransactionAmt must be positive, got {v}")
        return v

    @field_validator("TransactionDT")
    @classmethod
    def dt_must_be_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"TransactionDT must be non-negative, got {v}")
        return v


class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput] = Field(
        ...,
        description="List of transactions to score",
        min_length=1,
        max_length=10000,
    )

    @model_validator(mode="after")
    def check_transaction_ids_unique(self) -> "BatchTransactionInput":
        ids = [t.TransactionID for t in self.transactions]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "All TransactionIDs in a batch must be unique. Duplicate IDs detected."
            )
        return self


class PredictionOutput(BaseModel):
    TransactionID: int
    fraud_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's estimated probability this is fraud",
    )
    is_fraud: bool = Field(
        ...,
        description=(
            "True if fraud_probability >= threshold_used. This is the actionable decision."
        ),
    )
    risk_level: str = Field(
        ...,
        description="LOW / MEDIUM / HIGH / CRITICAL based on probability",
    )
    threshold_used: float = Field(
        ...,
        description="Decision threshold applied to fraud_probability",
    )
    model_version: str = Field(
        ...,
        description="MLflow model version that produced this prediction",
    )


class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
    total_transactions: int
    flagged_as_fraud: int
    fraud_rate_in_batch: float = Field(..., description="Fraction of batch flagged as fraud")
    model_version: str


class HealthResponse(BaseModel):
    """Response for the health check endpoint."""

    status: str
    model_loaded: bool
    model_version: str
    threshold: float
    redis_healthy: bool = False


def compute_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "LOW"
    elif probability < 0.5:
        return "MEDIUM"
    elif probability < 0.8:
        return "HIGH"
    else:
        return "CRITICAL"
