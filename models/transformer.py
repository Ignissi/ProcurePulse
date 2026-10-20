"""Признаки → токены → внимание между признаками → CLS-представление → две головы."""

import math

import torch
from torch import nn


class Tokenizer(nn.Module):
    def __init__(self, numeric_count, category_sizes, width=32):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(numeric_count, width) * 0.02)
        self.bias = nn.Parameter(torch.zeros(numeric_count, width))
        self.embeddings = nn.ModuleList()
        for size in category_sizes:
            self.embeddings.append(nn.Embedding(size, width))

    def forward(self, numeric, categories):
        numeric_tokens = numeric.unsqueeze(2) * self.weight + self.bias

        category_tokens = []
        for column, embedding in enumerate(self.embeddings):
            category_tokens.append(embedding(categories[:, column]))

        category_tokens = torch.stack(category_tokens, dim=1)
        return torch.cat([numeric_tokens, category_tokens], dim=1)


class Attention(nn.Module):
    def __init__(self, width=32, heads=4):
        super().__init__()
        if width % heads != 0:
            raise ValueError("Размер представления должен делиться на число голов внимания без остатка")

        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def split_heads(self, tensor):
        batch_size, token_count, width = tensor.shape
        tensor = tensor.reshape(batch_size, token_count, self.heads, self.head_width)
        return tensor.transpose(1, 2)

    def forward(self, tokens):
        batch_size, token_count, width = tokens.shape

        query = self.split_heads(self.query(tokens))
        key = self.split_heads(self.key(tokens))
        value = self.split_heads(self.value(tokens))

        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_width)
        attention = torch.softmax(scores, dim=-1)
        context = attention @ value

        context = context.transpose(1, 2).reshape(batch_size, token_count, width)
        return self.output(context)


class TransformerBlock(nn.Module):
    def __init__(self, width=32, heads=4, dropout=0.1):
        super().__init__()

        self.attention_norm = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        normalized = self.attention_norm(tokens)
        tokens = tokens + self.dropout(self.attention(normalized))

        normalized = self.feedforward_norm(tokens)
        tokens = tokens + self.dropout(self.feedforward(normalized))
        return tokens


class Transformer(nn.Module):
    def __init__(self, numeric_count, category_sizes, width=32, heads=4, dropout=0.1):
        super().__init__()

        self.tokenizer = Tokenizer(numeric_count, category_sizes, width)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.block1 = TransformerBlock(width, heads, dropout)
        self.block2 = TransformerBlock(width, heads, dropout)
        self.final_norm = nn.LayerNorm(width)
        self.price_head = nn.Linear(width, 1)
        self.outcome_head = nn.Linear(width, 1)

    def forward(self, numeric, categories):
        tokens = self.tokenizer(numeric, categories)
        cls = self.cls.expand(len(tokens), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        tokens = self.block1(tokens)
        tokens = self.block2(tokens)
        representation = self.final_norm(tokens)[:, 0]

        price = self.price_head(representation).squeeze(1)
        outcome_logit = self.outcome_head(representation).squeeze(1)
        return price, outcome_logit
