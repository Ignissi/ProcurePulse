"""Считает историю заказчика, поставщика и их пары только по прошлым датам."""

import numpy as np


def add_count(contracts, keys, column_name):
    grouping = keys + ["sign_date"]
    daily = contracts.groupby(grouping).size().rename("contracts_today").reset_index()
    daily = daily.sort_values("sign_date")

    total_so_far = daily.groupby(keys)["contracts_today"].cumsum()
    daily[column_name] = total_so_far - daily["contracts_today"]

    columns = grouping + [column_name]
    result = contracts.merge(daily[columns], on=grouping, how="left")

    for key in keys:
        result.loc[result[key] == "", column_name] = np.nan

    return result


def build_features(contracts):
    features = add_count(contracts, ["customer_inn"], "buyer_contracts")
    features = add_count(features, ["supplier_inn"], "supplier_contracts")
    features = add_count(features, ["customer_inn", "supplier_inn"], "pair_contracts")

    buyer_history = features["buyer_contracts"].replace(0, np.nan)
    features["pair_share"] = features["pair_contracts"] / buyer_history

    features = features.sort_values(["sign_date", "contract_id"])
    return features.reset_index(drop=True)
