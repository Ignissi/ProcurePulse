"""Строит две цели v1: контекстную цену и просрочку поставщика."""

import math

import pandas as pd

from config import HORIZON_DAYS


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def positive_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number) or number <= 0:
        return None
    return number


def get_outcome(contract, sign_date, observed_until):
    window_end = sign_date + pd.Timedelta(days=HORIZON_DAYS)
    if window_end > observed_until:
        return None

    procedures = as_list(contract.get("contractProcedure"))
    for procedure in procedures:
        containers = as_list(procedure.get("penalties"))
        penalties = []

        for container in containers:
            accruals = container.get("penaltyAccrual", container)
            penalties.extend(as_list(accruals))

        for penalty in penalties:
            reason = penalty.get("penaltyReason") or {}
            document = penalty.get("penaltyDocument") or {}
            date_text = str(document.get("documentDate", ""))[:10]
            event_date = pd.to_datetime(date_text, format="%Y-%m-%d", errors="coerce")

            supplier_penalty = penalty.get("contractParty") == "S"
            delay_reason = str(reason.get("code")) == "21"
            inside_window = sign_date < event_date <= window_end

            if supplier_penalty and delay_reason and inside_window:
                return 1

    return 0


def price_rejection(contract, products):
    if len(products) != 1:
        return "several_items"

    item = products[0]
    classification = item.get("OKPD2") or {}
    unit = item.get("OKEI") or {}
    currency = contract.get("currency") or {}
    code = str(classification.get("code", ""))[:2]

    if not code.isdigit():
        return "missing_product_code"
    if not 1 <= int(code) <= 32:
        return "service_or_work"
    if str(unit.get("code", "")) in ["", "876"]:
        return "unknown_or_conditional_unit"
    if currency.get("code") != "RUB":
        return "other_currency"
    if not item.get("name"):
        return "missing_description"

    quantity = positive_number(item.get("quantity"))
    price = positive_number(item.get("price"))
    total = positive_number(item.get("sum"))
    if quantity is None or price is None or total is None:
        return "missing_quantity_price_or_sum"

    calculated_total = quantity * price
    if not math.isclose(calculated_total, total, rel_tol=0.02, abs_tol=1):
        return "amount_mismatch"

    return "accepted"
