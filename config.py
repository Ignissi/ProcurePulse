"""Пути, временные границы, признаки и параметры обучения."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FEATURES_FILE = PROCESSED / "features.parquet"
RESULTS = ROOT / "results"

MONTHS = ["202210", "202211", "202212", "202301", "202302", "202303", "202304"]
# Дата, на которую зафиксированы исходные архивы.
OBSERVED_UNTIL = "2024-06-04"
VALIDATION_START = "2023-03-01"
TEST_START = "2023-04-01"
HORIZON_DAYS = 365

SEED = 42
EPOCHS = 8
BATCH_SIZE = 512
LEARNING_RATE = 0.001
MIN_PRICE_EXAMPLES = 5

# Текущая цена и сумма контракта не входят в признаки модели.
NUMERIC_COLUMNS = [
    "quantity",
    "item_count",
    "supplier_count",
    "month",
    "buyer_contracts",
    "supplier_contracts",
    "pair_contracts",
    "pair_share",
]

CATEGORY_COLUMNS = [
    "customer_inn",
    "supplier_inn",
    "okpd2_code",
    "unit_code",
    "product_key",
]

DATABASE_URL = "postgresql+psycopg://procurepulse:procurepulse@localhost:5433/procurepulse"
