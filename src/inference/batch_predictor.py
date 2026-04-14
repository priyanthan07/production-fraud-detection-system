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
        Score a single chunk of transactions.
    """
    
    chunk = chunk.copy()
 
    # Run feature engineering
    chunk = predictor._run_feature_pipeline(chunk)
 
    # Select model features
    X = predictor._select_model_features(chunk)
 
    # Score
    fraud_probabilities = predictor.model.predict_proba(X)[:, 1]
 
    # Build results DataFrame
    results = pd.DataFrame({
        "TransactionID": chunk["TransactionID"].values
        if "TransactionID" in chunk.columns
        else range(len(chunk)),
        "fraud_probability": fraud_probabilities.round(6),
        "is_fraud": fraud_probabilities >= predictor.threshold,
        "risk_level": [
            compute_risk_level(p) for p in fraud_probabilities
        ],
        "threshold_used": predictor.threshold,
        "model_version": str(predictor.model_version),
    })
 
    return results

def run_batch_scoring(
    input_path: str,
    output_path: str,
    identity_path: str = None,
    model_uri: str = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict:
    """ 
        Main batch scoring function.
    """
    
    start_time = time.time()
 
    # ----------------------------------------------------------------
    # Load predictor
    # ----------------------------------------------------------------
    p = FraudPredictor()
    p.load(model_uri=model_uri)
 
    # ----------------------------------------------------------------
    # Load input data
    # ----------------------------------------------------------------
    df = load_input_data(input_path, identity_path)
    total_rows = len(df)
    logger.info(f"Total rows to score: {total_rows:,}")
 
    # ----------------------------------------------------------------
    # Score in chunks
    # ----------------------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    all_results = []
    n_chunks = (total_rows + chunk_size - 1) // chunk_size
 
    for i in range(0, total_rows, chunk_size):
        chunk_num = i // chunk_size + 1
        chunk = df.iloc[i : i + chunk_size]
 
        logger.info(
            f"Scoring chunk {chunk_num}/{n_chunks} "
            f"(rows {i:,} to {min(i + chunk_size, total_rows):,})..."
        )
 
        try:
            chunk_results = score_chunk(p, chunk)
            all_results.append(chunk_results)
        except Exception as e:
            logger.error(
                f"Chunk {chunk_num} failed: {e}. "
                f"Skipping and continuing with next chunk."
            )
            # Write a placeholder with NaN scores for failed rows
            failed_results = pd.DataFrame({
                "TransactionID": chunk["TransactionID"].values
                if "TransactionID" in chunk.columns
                else range(i, min(i + chunk_size, total_rows)),
                "fraud_probability": np.nan,
                "is_fraud": None,
                "risk_level": "ERROR",
                "threshold_used": p.threshold,
                "model_version": str(p.model_version),
            })
            all_results.append(failed_results)
 
    # ----------------------------------------------------------------
    # Combine and save
    # ----------------------------------------------------------------
    results_df = pd.concat(all_results, ignore_index=True)
 
    if output_path.suffix == ".parquet":
        results_df.to_parquet(output_path, index=False)
    elif output_path.suffix == ".csv":
        results_df.to_csv(output_path, index=False)
    else:
        # Default to Parquet
        output_path = output_path.with_suffix(".parquet")
        results_df.to_parquet(output_path, index=False)
 
    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    elapsed = time.time() - start_time
    scored = results_df["fraud_probability"].notna().sum()
    flagged = results_df["is_fraud"].sum()
    errors = results_df["risk_level"].eq("ERROR").sum()
 
    summary = {
        "total_rows": total_rows,
        "successfully_scored": int(scored),
        "errors": int(errors),
        "flagged_as_fraud": int(flagged),
        "fraud_rate": round(float(flagged / scored) if scored > 0 else 0, 4),
        "elapsed_seconds": round(elapsed, 1),
        "rows_per_second": round(total_rows / elapsed, 0),
        "output_path": str(output_path),
        "model_version": str(p.model_version),
        "threshold": p.threshold,
    }
 
    logger.info("=" * 50)
    logger.info("Batch scoring complete.")
    logger.info(f"  Total rows:      {summary['total_rows']:,}")
    logger.info(f"  Scored:          {summary['successfully_scored']:,}")
    logger.info(f"  Errors:          {summary['errors']:,}")
    logger.info(f"  Flagged fraud:   {summary['flagged_as_fraud']:,}")
    logger.info(f"  Fraud rate:      {summary['fraud_rate']:.2%}")
    logger.info(f"  Time elapsed:    {summary['elapsed_seconds']}s")
    logger.info(f"  Throughput:      {summary['rows_per_second']:,.0f} rows/sec")
    logger.info(f"  Output:          {summary['output_path']}")
    logger.info("=" * 50)
 
    return summary


# CLI entry point
def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch score transactions for fraud detection."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV or Parquet file with transaction data.",
    )
    parser.add_argument(
        "--identity",
        default=None,
        help="Optional path to identity CSV or Parquet file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write predictions (Parquet or CSV).",
    )
    parser.add_argument(
        "--model-uri",
        default=None,
        help=(
            "MLflow model URI. Defaults to Production model. "
            "Example: 'models:/fraud_detection_model/3'"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows per processing chunk. Default: {DEFAULT_CHUNK_SIZE}",
    )
    return parser.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    run_batch_scoring(
        input_path=args.input,
        identity_path=args.identity,
        output_path=args.output,
        model_uri=args.model_uri,
        chunk_size=args.chunk_size,
    )
    
    