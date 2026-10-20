"""Готовит данные: делит их по времени, отбирает ценовые примеры и добавляет историю."""

from config import MIN_PRICE_EXAMPLES, PROCESSED, TEST_START, VALIDATION_START
from procurepulse.features import build_features
from procurepulse.read_data import read_contracts


def prepare():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    contracts, source_counts = read_contracts()

    contracts["split"] = "train"
    contracts.loc[contracts["sign_date"] >= VALIDATION_START, "split"] = "validation"
    contracts.loc[contracts["sign_date"] >= TEST_START, "split"] = "test"

    train_rows = contracts["split"] == "train"
    accepted_prices = contracts["price_reason"] == "accepted"
    price_train = contracts.loc[train_rows & accepted_prices]

    group_counts = price_train.groupby("product_key").size()
    group_support = contracts["product_key"].map(group_counts).fillna(0)
    contracts["price_mask"] = accepted_prices & (group_support >= MIN_PRICE_EXAMPLES)

    features = build_features(contracts)

    contracts.to_parquet(PROCESSED / "contracts.parquet", index=False, coerce_timestamps="us")
    features.to_parquet(PROCESSED / "features.parquet", index=False, coerce_timestamps="us")
    source_counts.to_csv(PROCESSED / "source_counts.csv", index=False)

    counts = contracts.groupby("split").agg(
        contracts=("contract_id", "size"),
        price_labels=("price_mask", "sum"),
        outcome_labels=("outcome_mask", "sum"),
        delays=("outcome", "sum"),
    )
    counts.to_csv(PROCESSED / "split_counts.csv")
    contracts["price_reason"].value_counts().to_csv(PROCESSED / "price_selection.csv")

    print(counts)
    return features


if __name__ == "__main__":
    prepare()
