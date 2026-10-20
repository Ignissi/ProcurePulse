"""Считает историю заказчика, поставщика и их пары через PySpark."""

import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from config import PROCESSED


def spark_features():
    builder = SparkSession.builder.master("local[2]").appName("ProcurePulse")
    builder = builder.config("spark.driver.host", "127.0.0.1")
    builder = builder.config("spark.driver.bindAddress", "127.0.0.1")
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    contracts = spark.read.parquet(str(PROCESSED / "contracts.parquet"))
    columns = ["contract_id", "customer_inn", "supplier_inn", "sign_date"]
    frame = contracts.select(columns)
    frame = frame.withColumn("day", F.datediff("sign_date", F.lit("1970-01-01")))

    buyer_window = Window.partitionBy("customer_inn").orderBy("day")
    buyer_window = buyer_window.rangeBetween(Window.unboundedPreceding, -1)
    supplier_window = Window.partitionBy("supplier_inn").orderBy("day")
    supplier_window = supplier_window.rangeBetween(Window.unboundedPreceding, -1)
    pair_window = Window.partitionBy("customer_inn", "supplier_inn").orderBy("day")
    pair_window = pair_window.rangeBetween(Window.unboundedPreceding, -1)

    buyer_known = F.col("customer_inn") != ""
    supplier_known = F.col("supplier_inn") != ""
    pair_known = buyer_known & supplier_known
    buyer_count = F.count("contract_id").over(buyer_window)
    supplier_count = F.count("contract_id").over(supplier_window)
    pair_count = F.count("contract_id").over(pair_window)
    frame = frame.withColumn("buyer_contracts", F.when(buyer_known, buyer_count))
    frame = frame.withColumn("supplier_contracts", F.when(supplier_known, supplier_count))
    frame = frame.withColumn("pair_contracts", F.when(pair_known, pair_count))
    frame = frame.withColumn(
        "pair_share",
        F.when(F.col("buyer_contracts") > 0, F.col("pair_contracts") / F.col("buyer_contracts")),
    )

    history_columns = ["contract_id", "buyer_contracts", "supplier_contracts", "pair_contracts", "pair_share"]
    history = frame.select(history_columns)
    features = contracts.join(history, on="contract_id", how="left")
    features.write.mode("overwrite").parquet(str(PROCESSED / "spark_features"))
    print("PySpark: строк с признаками:", features.count())
    spark.stop()

    # На выходе остаётся та же форма таблицы, что и в основном расчёте.
    result = pd.read_parquet(PROCESSED / "spark_features")
    result = result.sort_values(["sign_date", "contract_id"]).reset_index(drop=True)
    return result


if __name__ == "__main__":
    spark_features()
