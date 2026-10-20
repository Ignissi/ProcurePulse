"""Интерфейс Streamlit для просмотра очереди контрактов."""

import pandas as pd
import streamlit as st

from config import RESULTS


st.set_page_config(page_title="ProcurePulse", layout="wide")
st.title("ProcurePulse")
st.write("Контракты, которые стоит проверить в первую очередь")

queue_path = RESULTS / "queue.csv"
if not queue_path.exists():
    st.info("Очередь ещё не создана. Полный запуск: python run.py")
    st.stop()

queue = pd.read_csv(
    queue_path,
    dtype={
        "contract_id": str,
        "reg_num": str,
        "customer_inn": str,
        "supplier_inn": str,
    },
)

query = st.text_input("ИНН, номер контракта или товар")
if query:
    search_columns = ["contract_id", "reg_num", "customer_inn", "supplier_inn", "item_name"]
    matched = pd.Series(False, index=queue.index)

    for column in search_columns:
        matched |= queue[column].fillna("").str.contains(query, case=False, regex=False)

    queue = queue.loc[matched]

limit = st.slider("Количество контрактов", 10, 200, 50)
st.metric("Контрактов в очереди", len(queue))

columns = [
    "contract_id",
    "item_name",
    "customer_inn",
    "supplier_inn",
    "outcome_score",
    "pair_share",
    "price_score",
    "observed_price",
    "predicted_price",
    "price_ratio",
    "fusion_baseline",
    "audit_priority",
    "source_url",
]
st.dataframe(queue[columns].head(limit), use_container_width=True, hide_index=True)
st.caption("Основную очередь задаёт вероятность просрочки; история пары и ценовое отклонение показаны отдельно.")
st.caption("Объединённая оценка сохранена только для сравнения и не задаёт порядок очереди.")
st.caption("В v1 приоритет проверки равен вероятности просрочки.")
