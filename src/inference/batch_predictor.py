import argparse
import logging
import time
import pandas as pd
import numpy as np
from pathlib import Path

from src.inference.predictor import FraudPredictor
from src.inference.schemas import compute_risk_level

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_CHUNK_SIZE = 5000


def load_input_data(
    transaction_path: str,
    identity_path: str = None,
) -> pd.DataFrame:
    """
    Load transaction and optionally identity data from CSV or Parquet.
    Merges them the same way loader.py does during training.
    """

    path = Path(transaction_path)

    logger.info(f"Loading transactions from {path}...")
    if path.suffix == ".parquet":
        transactions = pd.read_parquet(path)
    else:
        transactions = pd.read_csv(path)

    logger.info(f"Loaded {len(transactions)} transactions.")

    if identity_path is not None:
        id_path = Path(identity_path)
        logger.info(f"Loading identity data from {id_path}...")
        if id_path.suffix == ".parquet":
            identity = pd.read_parquet(id_path)
        else:
            identity = pd.read_csv(id_path)

        logger.info(f"Loaded {len(identity)} identity records.")
        df = transactions.merge(identity, on="TransactionID", how="left")
        logger.info(
            f"Merged shape: {df.shape}. "
            f"Identity match rate: {identity['TransactionID'].isin(df['TransactionID']).mean():.2%}"
        )
    else:
        df = transactions

    return df


def score_chunk(
    predictor: FraudPredictor,
    chunk: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score a chunk using predict_single for each row.

    This uses the correct feature pipeline:
    - Stateless features computed locally
    - Velocity/aggregation features from Redis
    - Redis updated after each transaction
    - Scored features logged to disk

    Note: batch scoring via predict_single is slower than direct
    model.predict_proba() but ensures correct feature computation
    and Redis state management.
    """
    from src.inference.schemas import TransactionInput

    results = []

    # Sort by TransactionDT so velocity is computed correctly
    chunk = chunk.sort_values("TransactionDT").reset_index(drop=True)

    for _, row in chunk.iterrows():
        try:
            # Build TransactionInput from row — only pass known fields
            # Unknown fields (extra V columns etc.) are handled by extra="allow"
            row_dict = {
                k: (None if pd.isna(v) else v)
                for k, v in row.items()
            }

            txn = TransactionInput(**row_dict)
            result = predictor.predict_single(txn)

            results.append({
                "TransactionID":    result.TransactionID,
                "fraud_probability": result.fraud_probability,
                "is_fraud":         result.is_fraud,
                "risk_level":       result.risk_level,
                "threshold_used":   result.threshold_used,
                "model_version":    result.model_version,
            })

        except Exception as e:
            txn_id = row.get("TransactionID", "unknown")
            logger.error(f"Failed to score TransactionID={txn_id}: {e}")
            results.append({
                "TransactionID":    row.get("TransactionID"),
                "fraud_probability": np.nan,
                "is_fraud":         None,
                "risk_level":       "ERROR",
                "threshold_used":   predictor.threshold,
                "model_version":    str(predictor.model_version),
            })

    return pd.DataFrame(results)


def run_batch_scoring(
    input_path: str,
    output_path: str,
    identity_path: str = None,
    model_uri: str = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict:

    start_time = time.time()

    p = FraudPredictor()
    p.load(model_uri=model_uri)

    df = load_input_data(input_path, identity_path)
    total_rows = len(df)
    logger.info(f"Total rows to score: {total_rows:,}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    n_chunks = (total_rows + chunk_size - 1) // chunk_size

    for i in range(0, total_rows, chunk_size):
        chunk_num = i // chunk_size + 1
        chunk     = df.iloc[i : i + chunk_size]

        logger.info(
            f"Scoring chunk {chunk_num}/{n_chunks} "
            f"(rows {i:,} to {min(i + chunk_size, total_rows):,})..."
        )

        chunk_results = score_chunk(p, chunk)
        all_results.append(chunk_results)

    results_df = pd.concat(all_results, ignore_index=True)

    if output_path.suffix == ".parquet":
        results_df.to_parquet(output_path, index=False)
    else:
        results_df.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    scored  = results_df["fraud_probability"].notna().sum()
    flagged = results_df["is_fraud"].sum()
    errors  = results_df["risk_level"].eq("ERROR").sum()

    summary = {
        "total_rows":           total_rows,
        "successfully_scored":  int(scored),
        "errors":               int(errors),
        "flagged_as_fraud":     int(flagged),
        "fraud_rate":           round(float(flagged / scored) if scored > 0 else 0, 4),
        "elapsed_seconds":      round(elapsed, 1),
        "rows_per_second":      round(total_rows / elapsed, 0),
        "output_path":          str(output_path),
        "model_version":        str(p.model_version),
        "threshold":            p.threshold,
    }

    logger.info("=" * 50)
    logger.info("Batch scoring complete.")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 50)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch score transactions for fraud detection."
    )
    parser.add_argument("--input",      required=True)
    parser.add_argument("--identity",   default=None)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--model-uri",  default=None)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_batch_scoring(
        input_path    = args.input,
        identity_path = args.identity,
        output_path   = args.output,
        model_uri     = args.model_uri,
        chunk_size    = args.chunk_size,
    )
