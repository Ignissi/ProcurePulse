"""Обучает MLP и Transformer и выбирает модель по Average Precision на проверочной выборке."""

import copy
import json
import random

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, EPOCHS, FEATURES_FILE, LEARNING_RATE, NUMERIC_COLUMNS, PROCESSED, RESULTS, SEED
from models.mlp import MLP
from models.transformer import Transformer
from procurepulse.dataset import ContractDataset
from procurepulse.engine import predict_batches, train_epoch
from procurepulse.metrics import evaluate
from procurepulse.preprocessing import fit_preprocessing


def create_model(name, preprocessing):
    numeric_count = 2 * len(NUMERIC_COLUMNS)
    category_sizes = preprocessing["category_sizes"]

    if name == "mlp":
        return MLP(numeric_count, category_sizes)
    if name == "transformer":
        return Transformer(numeric_count, category_sizes)

    raise ValueError(f"Неизвестное имя модели: {name}")


def train_model(name, train, validation, preprocessing):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = ContractDataset(train, preprocessing)
    validation_dataset = ContractDataset(validation, preprocessing)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

    model = create_model(name, preprocessing).to(device)

    prior = float(train.loc[train["outcome_mask"], "outcome"].mean())
    with torch.no_grad():
        model.outcome_head.bias.fill_(np.log(prior / (1 - prior)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    best_score = -1.0
    best_epoch = 0
    best_weights = None
    history = []

    with mlflow.start_run(run_name=name):
        mlflow.log_params({
            "model": name,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "device": device,
        })
        mlflow.log_artifact(str(PROCESSED / "split_counts.csv"))
        mlflow.log_artifact(str(PROCESSED / "source_counts.csv"))

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            log_prices, probabilities = predict_batches(model, validation_loader, preprocessing, device)
            metrics = evaluate(validation, log_prices, probabilities)
            score = metrics["average_precision"]

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_weights = copy.deepcopy(model.state_dict())

            row = {"epoch": epoch, "train_loss": train_loss, **metrics}
            history.append(row)

            mlflow.log_metrics(metrics, step=epoch)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            print(name, "эпоха", epoch, "потери", round(train_loss, 4), "AP", round(score, 4), flush=True)

        model.load_state_dict(best_weights)
        log_prices, probabilities = predict_batches(model, validation_loader, preprocessing, device)
        metrics = evaluate(validation, log_prices, probabilities)

        price_rows = validation["price_mask"].to_numpy()
        actual_log_prices = np.log1p(validation.loc[price_rows, "observed_price"].to_numpy())
        residuals = actual_log_prices - log_prices[price_rows]
        residual_scale = max(float(np.quantile(np.abs(residuals), 0.9)), 0.1)

        model = model.cpu()
        bundle = {
            "model_name": name,
            "weights": model.state_dict(),
            "preprocessing": preprocessing,
            "train_prior": prior,
            "residual_scale": residual_scale,
            "best_epoch": best_epoch,
        }
        model_path = RESULTS / f"{name}.pt"
        history_path = RESULTS / f"{name}_history.csv"
        torch.save(bundle, model_path)
        pd.DataFrame(history).to_csv(history_path, index=False)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(history_path))

    return metrics


def train_models():
    RESULTS.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(FEATURES_FILE)
    train = frame.loc[frame["split"] == "train"].copy()
    validation = frame.loc[frame["split"] == "validation"].copy()

    counts = frame.groupby("split")[["price_mask", "outcome_mask"]].sum()
    if (counts == 0).any().any():
        raise ValueError("В одной из частей выборки нет меток; детали: data/processed/split_counts.csv")
    if train["outcome"].nunique() != 2 or validation["outcome"].nunique() != 2:
        raise ValueError("Для метрик классификации нужны оба класса в обучающей и проверочной выборках")

    preprocessing = fit_preprocessing(train)

    mlflow.set_tracking_uri("sqlite:///" + str(RESULTS / "mlflow.db"))
    mlflow.set_experiment("ProcurePulse")

    mlp_metrics = train_model("mlp", train, validation, preprocessing)
    transformer_metrics = train_model("transformer", train, validation, preprocessing)

    selected = "mlp"
    if transformer_metrics["average_precision"] > mlp_metrics["average_precision"]:
        selected = "transformer"

    report = {
        "selected": selected,
        "mlp_validation": mlp_metrics,
        "transformer_validation": transformer_metrics,
    }
    with (RESULTS / "selection.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("Выбрана модель:", selected)
    return selected


if __name__ == "__main__":
    train_models()
