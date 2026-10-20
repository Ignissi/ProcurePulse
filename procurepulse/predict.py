"""Получает прогнозы на тестовой выборке и строит итоговую очередь."""

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, FEATURES_FILE, RESULTS
from procurepulse.dataset import ContractDataset
from procurepulse.engine import predict_batches
from procurepulse.metrics import evaluate, ranking_metrics
from procurepulse.train import create_model


def make_queue(frame, log_prices, probabilities, residual_scale):
    queue = frame.copy()
    queue["outcome_score"] = probabilities
    queue["relation_score"] = queue["pair_share"]
    queue["predicted_price"] = np.nan
    queue["price_ratio"] = np.nan
    queue["price_score"] = np.nan

    price_rows = queue["price_mask"].to_numpy()
    predicted_prices = np.expm1(np.clip(log_prices[price_rows], 0, 30))
    actual_prices = queue.loc[price_rows, "observed_price"].to_numpy()

    queue.loc[price_rows, "predicted_price"] = predicted_prices
    queue.loc[price_rows, "price_ratio"] = actual_prices / np.maximum(predicted_prices, 0.01)

    residuals = np.log1p(actual_prices) - log_prices[price_rows]
    queue.loc[price_rows, "price_score"] = np.clip(residuals / residual_scale, 0, 1)

    # Фиксированное объединение сохранено только для сравнения.
    # Основную очередь задаёт outcome_score; цена и история пары остаются отдельными сигналами.
    weights = {
        "outcome_score": 0.60,
        "relation_score": 0.25,
        "price_score": 0.15,
    }
    weighted_sum = pd.Series(0.0, index=queue.index)
    available_weight = pd.Series(0.0, index=queue.index)

    for column, weight in weights.items():
        known = queue[column].notna()
        weighted_sum += queue[column].fillna(0) * weight
        available_weight += known * weight

    queue["fusion_baseline"] = weighted_sum / available_weight
    queue["audit_priority"] = queue["outcome_score"]

    base_url = "https://zakupki.gov.ru/epz/contract/contractCard/common-info.html?reestrNumber="
    queue["source_url"] = base_url + queue["reg_num"]

    return queue.sort_values(["audit_priority", "contract_id"], ascending=[False, True])


def predict():
    torch.set_num_threads(4)

    with (RESULTS / "selection.json").open(encoding="utf-8") as file:
        selection = json.load(file)

    model_name = selection["selected"]
    bundle = torch.load(RESULTS / f"{model_name}.pt", map_location="cpu", weights_only=True)
    preprocessing = bundle["preprocessing"]

    features = pd.read_parquet(FEATURES_FILE)
    test = features.loc[features["split"] == "test"].reset_index(drop=True)

    model = create_model(model_name, preprocessing)
    model.load_state_dict(bundle["weights"])

    dataset = ContractDataset(test, preprocessing)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    log_prices, probabilities = predict_batches(model, loader, preprocessing, "cpu")

    queue = make_queue(test, log_prices, probabilities, bundle["residual_scale"])
    queue.to_csv(RESULTS / "queue.csv", index=False)

    known_pairs = (queue["customer_inn"] != "") & (queue["supplier_inn"] != "")
    pairs = queue.loc[known_pairs].drop_duplicates(["customer_inn", "supplier_inn"])
    pairs.to_csv(RESULTS / "pairs.csv", index=False)

    train = features.loc[features["split"] == "train"]
    price_train = train.loc[train["price_mask"]]
    group_prices = price_train.groupby("product_key")["observed_price"].median()
    baseline_prices = test["product_key"].map(group_prices)
    baseline_prices = baseline_prices.fillna(price_train["observed_price"].median())
    baseline_log_prices = np.log1p(baseline_prices.to_numpy())
    baseline_probabilities = np.full(len(test), bundle["train_prior"])

    labelled_queue = queue.loc[queue["outcome_mask"]]
    labels = labelled_queue["outcome"].to_numpy()
    primary_priorities = labelled_queue["audit_priority"].to_numpy()
    fusion_priorities = labelled_queue["fusion_baseline"].to_numpy()

    report = {
        "selected": model_name,
        "model": evaluate(test, log_prices, probabilities),
        "baseline": evaluate(test, baseline_log_prices, baseline_probabilities),
        "ranking": {
            "primary_strategy": "outcome_score",
            "primary": ranking_metrics(labels, primary_priorities),
            "fusion_baseline": ranking_metrics(labels, fusion_priorities),
        },
    }
    with (RESULTS / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("Очередь построена. Контрактов:", len(queue))
    print(json.dumps(report, indent=2))
    return queue


if __name__ == "__main__":
    predict()
