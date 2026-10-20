"""Готовит признаки по статистикам обучающей выборки."""

import numpy as np
import torch

from config import CATEGORY_COLUMNS, NUMERIC_COLUMNS


def fit_preprocessing(train):
    numeric = np.log1p(train[NUMERIC_COLUMNS].astype(float))
    medians = numeric.median().fillna(0)
    numeric = numeric.fillna(medians)
    means = numeric.mean()
    scales = numeric.std(ddof=0).replace(0, 1)

    vocabularies = {}
    category_sizes = []

    for column in CATEGORY_COLUMNS:
        values = sorted(train[column].fillna("").unique())
        vocabulary = {"": 0}

        next_index = 2  # Индекс 1 — категория, которой не было в обучающей выборке.
        for value in values:
            if value != "":
                vocabulary[value] = next_index
                next_index += 1

        vocabularies[column] = vocabulary
        category_sizes.append(next_index)

    prices = train.loc[train["price_mask"], "observed_price"]
    log_prices = np.log1p(prices)

    return {
        "medians": medians.to_dict(),
        "means": means.to_dict(),
        "scales": scales.to_dict(),
        "vocabularies": vocabularies,
        "category_sizes": category_sizes,
        "price_mean": float(log_prices.mean()),
        "price_scale": max(float(log_prices.std(ddof=0)), 1.0),
    }


def encode_features(frame, preprocessing):
    numeric = np.log1p(frame[NUMERIC_COLUMNS].astype(float))
    missing = numeric.isna().astype(float)
    numeric = numeric.fillna(preprocessing["medians"])

    for column in NUMERIC_COLUMNS:
        mean = preprocessing["means"][column]
        scale = preprocessing["scales"][column]
        numeric[column] = (numeric[column] - mean) / scale

    numeric_values = np.concatenate([numeric.to_numpy(), missing.to_numpy()], axis=1)

    categories = []
    for column in CATEGORY_COLUMNS:
        vocabulary = preprocessing["vocabularies"][column]
        encoded = frame[column].fillna("").map(vocabulary).fillna(1)
        categories.append(encoded.to_numpy())

    category_values = np.stack(categories, axis=1)
    numeric_tensor = torch.tensor(numeric_values, dtype=torch.float32)
    category_tensor = torch.tensor(category_values, dtype=torch.long)
    return numeric_tensor, category_tensor
