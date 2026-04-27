from feast import FileSource

# Main training features — produced by the feature pipeline
# This is the offline store for all engineered features
train_features_source = FileSource(
    name="train_features_source",
    path="data/processed/train_features.parquet",
    timestamp_field="TransactionDT",
    description=(
        "Full feature-engineered training dataset. "
        "Produced by src/features/pipeline.py. "
        "Contains velocity, aggregation, time, and encoded categorical features."
    ),
)

# Production scored features — features logged at inference time
# Used after labels arrive to build new training data
scored_features_source = FileSource(
    name="scored_features_source",
    path="data/production/scored_features.parquet",
    timestamp_field="TransactionDT",
    description=(
        "Features computed at inference time for each production transaction. "
        "Joined with labels.parquet for retraining."
    ),
)
