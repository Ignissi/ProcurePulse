"""Обучает модель и получает прогнозы для MLP и Transformer."""

import numpy as np
import torch
from torch import nn


def train_epoch(model, loader, optimizer, device):
    model.train()
    price_loss_function = nn.SmoothL1Loss()
    outcome_loss_function = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    trained_batches = 0

    for batch in loader:
        numeric = batch["numeric"].to(device)
        categories = batch["categories"].to(device)
        price_target = batch["price"].to(device)
        outcome_target = batch["outcome"].to(device)
        price_mask = batch["price_mask"].to(device)
        outcome_mask = batch["outcome_mask"].to(device)

        optimizer.zero_grad()
        price, outcome_logit = model(numeric, categories)

        losses = []
        if price_mask.any():
            losses.append(price_loss_function(price[price_mask], price_target[price_mask]))
        if outcome_mask.any():
            losses.append(outcome_loss_function(outcome_logit[outcome_mask], outcome_target[outcome_mask]))
        if not losses:
            continue

        loss = sum(losses)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        trained_batches += 1

    return total_loss / max(trained_batches, 1)


def predict_batches(model, loader, preprocessing, device):
    model.eval()
    all_prices = []
    all_probabilities = []

    with torch.no_grad():
        for batch in loader:
            numeric = batch["numeric"].to(device)
            categories = batch["categories"].to(device)
            price, outcome_logit = model(numeric, categories)

            log_price = price * preprocessing["price_scale"] + preprocessing["price_mean"]
            probability = torch.sigmoid(outcome_logit)

            all_prices.append(log_price.cpu().numpy())
            all_probabilities.append(probability.cpu().numpy())

    return np.concatenate(all_prices), np.concatenate(all_probabilities)
