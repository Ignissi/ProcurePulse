"""Базовая MLP: числовые признаки и эмбеддинги → общее представление → две головы."""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, numeric_count, category_sizes, embedding_dim=8, hidden_dim=64, dropout=0.1):
        super().__init__()

        self.embeddings = nn.ModuleList()
        for size in category_sizes:
            self.embeddings.append(nn.Embedding(size, embedding_dim))

        input_size = numeric_count + embedding_dim * len(category_sizes)
        representation_size = hidden_dim // 2

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, representation_size),
            nn.ReLU(),
        )
        self.price_head = nn.Linear(representation_size, 1)
        self.outcome_head = nn.Linear(representation_size, 1)

    def forward(self, numeric, categories):
        parts = [numeric]
        for column, embedding in enumerate(self.embeddings):
            parts.append(embedding(categories[:, column]))

        features = torch.cat(parts, dim=1)
        representation = self.encoder(features)

        price = self.price_head(representation).squeeze(1)
        outcome_logit = self.outcome_head(representation).squeeze(1)
        return price, outcome_logit
