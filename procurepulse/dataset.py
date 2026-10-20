"""Преобразует строки таблицы признаков в тензоры для DataLoader."""

import numpy as np
import torch
from torch.utils.data import Dataset

from procurepulse.preprocessing import encode_features


class ContractDataset(Dataset):
    def __init__(self, frame, preprocessing):
        self.numeric, self.categories = encode_features(frame, preprocessing)

        price = np.log1p(frame["observed_price"].fillna(0))
        price = (price - preprocessing["price_mean"]) / preprocessing["price_scale"]
        outcome = frame["outcome"].fillna(0)

        self.price = torch.tensor(price.to_numpy(), dtype=torch.float32)
        self.outcome = torch.tensor(outcome.to_numpy(), dtype=torch.float32)
        self.price_mask = torch.tensor(frame["price_mask"].to_numpy(), dtype=torch.bool)
        self.outcome_mask = torch.tensor(frame["outcome_mask"].to_numpy(), dtype=torch.bool)

    def __len__(self):
        return len(self.price)

    def __getitem__(self, index):
        return {
            "numeric": self.numeric[index],
            "categories": self.categories[index],
            "price": self.price[index],
            "outcome": self.outcome[index],
            "price_mask": self.price_mask[index],
            "outcome_mask": self.outcome_mask[index],
        }
