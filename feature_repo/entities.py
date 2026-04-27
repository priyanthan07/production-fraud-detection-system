from feast import Entity
from feast.types import Int64, String

card_entity = Entity(
    name="card",
    join_keys=["card1"],
    value_type=Int64,
    description="Card identifier (card1 column). Primary entity for per-card features.",
)

email_entity = Entity(
    name="email",
    join_keys=["P_emaildomain"],
    value_type=String,
    description="Purchaser email domain. Primary entity for per-email features.",
)
