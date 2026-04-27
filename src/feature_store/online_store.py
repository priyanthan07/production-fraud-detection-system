import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import redis

logger = logging.getLogger(__name__)

# Time windows in seconds — must match velocity_features.py exactly
WINDOW_1HR  = 3600
WINDOW_24HR = 86400
WINDOW_7DAY = 604800

# Redis key templates
_CARD_TS_KEY    = "card:{card_id}:timestamps"
_CARD_AMT_KEY   = "card:{card_id}:amounts"
_CARD_STAT_KEY  = "card:{card_id}:stats"
_EMAIL_TS_KEY   = "email:{email}:timestamps"
_EMAIL_AMT_KEY  = "email:{email}:amounts"
_EMAIL_STAT_KEY = "email:{email}:stats"

# Keys expire after 90 days of inactivity
_KEY_TTL = 90 * 86400

# Output paths
_SCORED_FEATURES_PATH     = Path("data/production/scored_features.parquet")
_RAW_TRANSACTIONS_PATH    = Path("data/production/raw_transactions.parquet")
_RECENT_PREDICTIONS_PATH  = Path("data/production/recent_predictions.parquet")

# Rolling window for drift detection — keep last 30 days only
_DRIFT_WINDOW_SECONDS = 30 * 86400

def _safe_email(email: str) -> str:
    """Sanitise email domain for use as a Redis key."""
    return email.replace(":", "_").replace(" ", "_").replace("/", "_")

class OnlineFeatureStore:
    """
        Redis-backed online feature store for real-time fraud detection.

        Usage:
            store = OnlineFeatureStore()
            store.connect()

            # At inference — BEFORE scoring:
            card_feats  = store.get_card_features(card_id, txn_time, txn_amt)
            email_feats = store.get_email_features(email, txn_time, txn_amt)

            # Score model with features ...

            # At inference — AFTER scoring:
            store.update(txn_id, card_id, email, amount, txn_time)
            store.log_scored_features(txn_id, card_id, email, amount, txn_time, features, fraud_prob, is_fraud)
            store.log_raw_transaction(raw_dict)
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
    ):
        self.host     = host
        self.port     = port
        self.db       = db
        self.password = password
        self._client: Optional[redis.Redis] = None
        
    def connect(self) -> None:
        """Connect to Redis. Raises redis.ConnectionError on failure."""
        self._client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=2,
        )
        self._client.ping()
        logger.info(f"OnlineFeatureStore connected to Redis at {self.host}:{self.port}")

    def is_healthy(self) -> bool:
        """Return True if Redis connection is alive."""
        try:
            return self._client is not None and bool(self._client.ping())
        except redis.RedisError:
            return False

    @property
    def _c(self) -> redis.Redis:
        """Internal: return client or raise if not connected."""
        if self._client is None:
            raise RuntimeError("OnlineFeatureStore is not connected. Call connect() first.")
        return self._client

    def get_card_features(
        self,
        card_id: Optional[int],
        current_time: float,
        current_amt: float,
    ) -> dict:
        """
            Fetch velocity and aggregation features for card_id.

            Only transactions STRICTLY BEFORE current_time are used.
            current_amt is the amount being scored — not included in history.
            This mirrors the shift(1) + expanding() pattern used in training.

            Returns a dict of feature_name → value.
            All keys match exactly the column names in train_features.parquet.
        """
        if card_id is None:
            return self._null_card_features()

        try:
            return self._fetch_entity_features(
                ts_key   = _CARD_TS_KEY.format(card_id=card_id),
                amt_key  = _CARD_AMT_KEY.format(card_id=card_id),
                stat_key = _CARD_STAT_KEY.format(card_id=card_id),
                prefix   = "card1",
                current_time = current_time,
                current_amt  = current_amt,
                include_time_features=True,
            )
        except redis.RedisError as e:
            logger.error(f"Redis error fetching card features for card_id={card_id}: {e}")
            return self._null_card_features()
        
    def get_email_features(
        self,
        email_domain: Optional[str],
        current_time: float,
        current_amt: float,
    ) -> dict:
        """
            Fetch velocity and aggregation features for email_domain.
            Same structure as card features.
        """
        if not email_domain:
            return self._null_email_features()

        safe = _safe_email(email_domain)

        try:
            return self._fetch_entity_features(
                ts_key   = _EMAIL_TS_KEY.format(email=safe),
                amt_key  = _EMAIL_AMT_KEY.format(email=safe),
                stat_key = _EMAIL_STAT_KEY.format(email=safe),
                prefix   = "P_emaildomain",
                current_time = current_time,
                current_amt  = current_amt,
                include_time_features=False,
            )
        except redis.RedisError as e:
            logger.error(
                f"Redis error fetching email features for domain={email_domain}: {e}"
            )
            return self._null_email_features()
        
    def _fetch_entity_features(
        self,
        ts_key: str,
        amt_key: str,
        stat_key: str,
        prefix: str,
        current_time: float,
        current_amt: float,
        include_time_features: bool,
    ) -> dict:
        """
            Shared implementation for card and email feature lookup.

            prefix = "card1"        → produces card1_count_1hr etc.
            prefix = "P_emaildomain" → produces P_emaildomain_count_1hr etc.
        """
        features = {}

        # ── Velocity features ──────────────────────────────────────
        # ZRANGEBYSCORE with "(" prefix means strictly less than
        for window_name, window_seconds in [
            ("1hr",  WINDOW_1HR),
            ("24hr", WINDOW_24HR),
            ("7day", WINDOW_7DAY),
        ]:
            window_start = current_time - window_seconds

            txn_ids = self._c.zrangebyscore(
                ts_key,
                window_start,
                f"({current_time}",
            )

            count   = len(txn_ids)
            amt_sum = 0.0

            if count > 0:
                raw_amts = self._c.hmget(amt_key, txn_ids)
                amt_sum = sum(float(a) for a in raw_amts if a is not None)

            features[f"{prefix}_count_{window_name}"]   = count
            features[f"{prefix}_amt_sum_{window_name}"] = amt_sum

        # ── Aggregation features ───────────────────────────────────
        stats = self._c.hgetall(stat_key)

        if stats:
            txn_count  = int(float(stats.get("txn_count",  0)))
            amt_sum_total = float(stats.get("amt_sum",   0.0))
            amt_sum_sq    = float(stats.get("amt_sum_sq", 0.0))
            first_seen    = float(stats.get("first_seen", current_time))
            last_seen     = float(stats.get("last_seen",  current_time))

            if txn_count > 0:
                mean = amt_sum_total / txn_count

                # Online variance — Welford's formula
                # var = (sum_sq - n * mean^2) / max(n-1, 1)
                variance = max(
                    (amt_sum_sq - txn_count * mean ** 2) / max(txn_count - 1, 1),
                    0.0,
                )
                std = variance ** 0.5

                deviation = current_amt - mean
                zscore    = deviation / std if std > 0.0 else 0.0

                if prefix == "card1":
                    features["card1_txn_count"]     = txn_count
                    features["card1_amt_mean"]      = mean
                    features["card1_amt_std"]       = std
                    features["card1_amt_deviation"] = deviation
                    features["card1_amt_zscore"]    = zscore

                    if include_time_features:
                        features["days_since_card_first_seen"] = (
                            (current_time - first_seen) / 86400
                        )
                        # time_since_last_txn_card1 matches fillna(-1) in training
                        features["time_since_last_txn_card1"] = (
                            self._time_since_last(ts_key, current_time)
                        )
                else:
                    # email prefix
                    features["email_amt_mean"]      = mean
                    features["email_amt_std"]       = std
                    features["email_amt_deviation"] = deviation
                    features["email_amt_zscore"]    = zscore
            else:
                features.update(self._null_aggregations(prefix, current_amt, include_time_features))
        else:
            # First time seeing this entity
            features.update(self._null_aggregations(prefix, current_amt, include_time_features))

        return features
    
    def _time_since_last(self, ts_key: str, current_time: float) -> float:
        """
            Seconds since the most recent past transaction.
            Returns -1.0 if no past transactions — matches fillna(-1) in training.
        """
        result = self._c.zrevrangebyscore(
            ts_key,
            f"({current_time}",
            "-inf",
            start=0,
            num=1,
            withscores=True,
        )
        if result:
            _, last_score = result[0]
            return float(current_time) - float(last_score)
        return -1.0
    
    def update(
        self,
        transaction_id: int,
        card_id: Optional[int],
        email_domain: Optional[str],
        amount: float,
        transaction_time: float,
    ) -> None:
        """
            Update Redis with a new transaction.

            MUST be called AFTER the prediction is returned to the caller.
            This ensures the current transaction does not influence its own
            features — identical to the shift(1) pattern used in training.

            Uses a pipeline for atomic multi-command execution.
        """
        txn_id_str = str(transaction_id)
        cutoff     = transaction_time - WINDOW_7DAY

        try:
            pipe = self._c.pipeline()

            if card_id is not None:
                ck_ts   = _CARD_TS_KEY.format(card_id=card_id)
                ck_amt  = _CARD_AMT_KEY.format(card_id=card_id)
                ck_stat = _CARD_STAT_KEY.format(card_id=card_id)

                pipe.zadd(ck_ts, {txn_id_str: transaction_time})
                pipe.hset(ck_amt, txn_id_str, amount)
                pipe.hincrbyfloat(ck_stat, "amt_sum",    amount)
                pipe.hincrbyfloat(ck_stat, "amt_sum_sq", amount ** 2)
                pipe.hincrby(ck_stat, "txn_count", 1)
                # NX = only set if field does Not eXist
                pipe.hsetnx(ck_stat, "first_seen", transaction_time)
                pipe.hset(ck_stat, "last_seen", transaction_time)
                # Keep only last 7 days of timestamps to bound memory
                pipe.zremrangebyscore(ck_ts, 0, cutoff)
                pipe.expire(ck_ts,   _KEY_TTL)
                pipe.expire(ck_amt,  _KEY_TTL)
                pipe.expire(ck_stat, _KEY_TTL)

            if email_domain:
                safe = _safe_email(email_domain)
                ek_ts   = _EMAIL_TS_KEY.format(email=safe)
                ek_amt  = _EMAIL_AMT_KEY.format(email=safe)
                ek_stat = _EMAIL_STAT_KEY.format(email=safe)

                pipe.zadd(ek_ts, {txn_id_str: transaction_time})
                pipe.hset(ek_amt, txn_id_str, amount)
                pipe.hincrbyfloat(ek_stat, "amt_sum",    amount)
                pipe.hincrbyfloat(ek_stat, "amt_sum_sq", amount ** 2)
                pipe.hincrby(ek_stat, "txn_count", 1)
                pipe.hsetnx(ek_stat, "first_seen", transaction_time)
                pipe.hset(ek_stat, "last_seen", transaction_time)
                pipe.zremrangebyscore(ek_ts, 0, cutoff)
                pipe.expire(ek_ts,   _KEY_TTL)
                pipe.expire(ek_amt,  _KEY_TTL)
                pipe.expire(ek_stat, _KEY_TTL)

            pipe.execute()

        except redis.RedisError as e:
            logger.error(
                f"Redis update failed for txn_id={transaction_id}: {e}. Prediction already returned. Feature store may be slightly stale."
            )
            
    def log_scored_features(
        self,
        transaction_id: int,
        card_id: Optional[int],
        email_domain: Optional[str],
        amount: float,
        transaction_time: float,
        features: dict,
        fraud_probability: float,
        is_fraud: bool,
        model_version: str,
    ) -> None:
        """
            Persist the feature values that were used to score this transaction.

            This is the critical data needed for future retraining.
            When labels arrive later, they are joined with this file on TransactionID.
            The resulting labeled dataset is merged with original training data
            for the next retraining run.

            Written to: data/production/scored_features.parquet
        """
        row = {
            "TransactionID":    transaction_id,
            "card1":            card_id,
            "P_emaildomain":    email_domain,
            "TransactionAmt":   amount,
            "TransactionDT":    transaction_time,
            "fraud_probability": round(fraud_probability, 6),
            "is_fraud":         is_fraud,
            "model_version":    model_version,
            "scored_at":        time.time(),
            **features,
        }
        self._append_to_parquet(row, _SCORED_FEATURES_PATH)

    def log_raw_transaction(self, raw_dict: dict) -> None:
        """
            Persist the raw incoming transaction fields for audit purposes.

            Written to: data/production/raw_transactions.parquet
            This file is never used for training — it is the audit log.
        """
        self._append_to_parquet(raw_dict, _RAW_TRANSACTIONS_PATH)

    def log_prediction_for_drift(
        self,
        transaction_id: int,
        transaction_time: float,
        features: dict,
    ) -> None:
        """
            Append a row to the rolling drift detection window.

            Written to: data/production/recent_predictions.parquet
            Read by:    drift_detector.py

            Keeps only the last 30 days of predictions to control file size.
            Only monitored features are stored — not all features.
        """
        row = {
            "TransactionID": transaction_id,
            "TransactionDT": transaction_time,
            **features,
        }

        _RECENT_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

        new_df = pd.DataFrame([row])

        if _RECENT_PREDICTIONS_PATH.exists():
            existing = pd.read_parquet(_RECENT_PREDICTIONS_PATH)
            combined = pd.concat([existing, new_df], ignore_index=True)

            # Rolling window: drop rows older than 30 days
            if "TransactionDT" in combined.columns:
                max_dt  = combined["TransactionDT"].max()
                cutoff  = max_dt - _DRIFT_WINDOW_SECONDS
                combined = combined[combined["TransactionDT"] >= cutoff]
        else:
            combined = new_df

        combined.to_parquet(_RECENT_PREDICTIONS_PATH, index=False)
        
    def bootstrap_from_parquet(
        self,
        parquet_path: str,
        batch_size: int = 5000,
    ) -> None:
        """
        Populate Redis from the historical training feature file.

        Run ONCE on day 1 after first training.
        Do NOT re-run after retraining — Redis already contains live state.

        Loads only:
          - Running stats (count, sum, sum_sq, first_seen, last_seen)
          - Last 7 days of timestamps (for velocity computation)

        Does NOT load all 590K rows into Redis memory.
        Only the aggregated statistics per card and email domain.
        """
        path = Path(parquet_path)

        if not path.exists():
            logger.warning(
                f"Bootstrap parquet not found at {path}. "
                f"Redis starts empty. First predictions will have zero velocity features."
            )
            return

        logger.info(f"Bootstrapping Redis from {path}...")

        # Load only the columns needed
        needed_cols = ["TransactionID", "card1", "P_emaildomain",
                       "TransactionAmt", "TransactionDT"]

        df = pd.read_parquet(path, columns=needed_cols)
        df = df.sort_values("TransactionDT").reset_index(drop=True)

        total     = len(df)
        processed = 0

        for start in range(0, total, batch_size):
            batch = df.iloc[start : start + batch_size]

            for _, row in batch.iterrows():
                card_id = int(row["card1"]) if pd.notna(row["card1"]) else None
                email   = row["P_emaildomain"] if pd.notna(row["P_emaildomain"]) else None
                amount  = float(row["TransactionAmt"])
                txn_dt  = float(row["TransactionDT"])
                txn_id  = int(row["TransactionID"])

                self.update(
                    transaction_id   = txn_id,
                    card_id          = card_id,
                    email_domain     = email,
                    amount           = amount,
                    transaction_time = txn_dt,
                )

            processed += len(batch)
            if processed % 50000 == 0 or processed == total:
                logger.info(f"Bootstrap: {processed:,}/{total:,} rows processed")

        logger.info(f"Bootstrap complete. {total:,} transactions loaded into Redis.")
        
    @staticmethod
    def _append_to_parquet(row: dict, path: Path) -> None:
        """Append a single row dict to a parquet file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame([row])

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        combined.to_parquet(path, index=False)

    def _null_card_features(self) -> dict:
        """Return zero/NaN features when card_id is None or Redis is down."""
        features = {}
        for w in ["1hr", "24hr", "7day"]:
            features[f"card1_count_{w}"]   = 0
            features[f"card1_amt_sum_{w}"] = 0.0
        features.update(self._null_aggregations("card1", 0.0, True))
        return features

    def _null_email_features(self) -> dict:
        """Return zero/NaN features when email_domain is None or Redis is down."""
        features = {}
        for w in ["1hr", "24hr", "7day"]:
            features[f"P_emaildomain_count_{w}"]   = 0
            features[f"P_emaildomain_amt_sum_{w}"] = 0.0
        features.update(self._null_aggregations("P_emaildomain", 0.0, False))
        return features

    @staticmethod
    def _null_aggregations(
        prefix: str,
        current_amt: float,
        include_time_features: bool,
    ) -> dict:
        """
        NaN/zero aggregations for first-time entities.
        Matches training behavior: first transaction per card has NaN mean.
        """
        if prefix == "card1":
            result = {
                "card1_txn_count":     0,
                "card1_amt_mean":      float("nan"),
                "card1_amt_std":       float("nan"),
                "card1_amt_deviation": float("nan"),
                "card1_amt_zscore":    0.0,
            }
            if include_time_features:
                result["days_since_card_first_seen"] = 0.0
                result["time_since_last_txn_card1"]  = -1.0
            return result
        else:
            return {
                "email_amt_mean":      float("nan"),
                "email_amt_std":       float("nan"),
                "email_amt_deviation": float("nan"),
                "email_amt_zscore":    0.0,
            }


# Module-level singleton used by predictor.py
feature_store = OnlineFeatureStore()
