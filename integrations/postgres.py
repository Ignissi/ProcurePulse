"""Сохраняет контракты, признаки и прогнозы в PostgreSQL."""

import pandas as pd
from sqlalchemy import create_engine

from config import DATABASE_URL, PROCESSED, RESULTS


def load_database():
    contracts = pd.read_parquet(PROCESSED / "contracts.parquet")
    features = pd.read_parquet(PROCESSED / "features.parquet")
    queue = pd.read_csv(RESULTS / "queue.csv", dtype={"contract_id": str})
    engine = create_engine(DATABASE_URL)

    contract_columns = [
        "contract_id",
        "reg_num",
        "sign_date",
        "customer_inn",
        "supplier_inn",
        "item_name",
        "okpd2_code",
        "unit_code",
        "quantity",
        "observed_price",
    ]
    feature_columns = ["contract_id", "buyer_contracts", "supplier_contracts", "pair_contracts", "pair_share"]
    prediction_columns = [
        "contract_id",
        "outcome_score",
        "price_score",
        "predicted_price",
        "price_ratio",
        "fusion_baseline",
        "audit_priority",
    ]

    sql_options = {"if_exists": "replace", "index": False, "chunksize": 1000, "method": "multi"}
    with engine.begin() as connection:
        contracts[contract_columns].to_sql("contracts", connection, **sql_options)
        features[feature_columns].to_sql("features", connection, **sql_options)
        queue[prediction_columns].to_sql("predictions", connection, **sql_options)

    print("PostgreSQL:", len(contracts), "контрактов;", len(queue), "прогнозов")
    engine.dispose()


if __name__ == "__main__":
    load_database()
