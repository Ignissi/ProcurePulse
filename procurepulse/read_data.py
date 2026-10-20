"""Читает архивы Clearspending и собирает таблицу московских контрактов по 44-ФЗ."""

import json
from zipfile import ZipFile

import pandas as pd

from config import MONTHS, OBSERVED_UNTIL, RAW
from procurepulse.targets import as_list, get_outcome, positive_number, price_rejection


def contract_to_row(contract, filename):
    buyer = contract.get("customer") or {}
    products = as_list(contract.get("products"))
    suppliers = as_list(contract.get("suppliers"))

    item = products[0] if len(products) == 1 else {}
    supplier = suppliers[0] if len(suppliers) == 1 else {}

    date_text = str(contract.get("signDate", ""))[:10]
    sign_date = pd.to_datetime(date_text, format="%Y-%m-%d", errors="coerce")
    if pd.isna(sign_date):
        return None

    classification = item.get("OKPD2") or {}
    unit = item.get("OKEI") or {}
    product_code = str(classification.get("code", ""))
    unit_code = str(unit.get("code", ""))

    description = str(item.get("name") or "").lower()
    description = " ".join(description.split())

    product_key = ""
    if description:
        product_key = product_code + " | " + unit_code + " | " + description

    outcome = get_outcome(contract, sign_date, pd.Timestamp(OBSERVED_UNTIL))

    return {
        "contract_id": str(contract["id"]),
        "reg_num": str(contract.get("regNum") or ""),
        "sign_date": sign_date,
        "published_at": str(contract.get("publishDate") or ""),
        "customer_inn": str(buyer.get("inn") or ""),
        "supplier_inn": str(supplier.get("inn") or ""),
        "item_name": description,
        "okpd2_code": product_code,
        "unit_code": unit_code,
        "product_key": product_key,
        "quantity": positive_number(item.get("quantity")),
        "item_count": len(products),
        "supplier_count": len(suppliers),
        "month": sign_date.month,
        "observed_price": positive_number(item.get("price")),
        "price_reason": price_rejection(contract, products),
        "outcome": outcome,
        "outcome_mask": outcome is not None,
        "source_file": filename,
    }


def read_contracts():
    rows = []
    file_counts = []

    for month in MONTHS:
        filename = "contracts_44fz_" + month + "-latest.jsonl.zip"
        path = RAW / filename
        print("Файл:", filename, flush=True)

        total = 0
        selected = 0

        with ZipFile(path) as archive:
            for member in archive.namelist():
                if not member.endswith(".jsonl"):
                    continue

                with archive.open(member) as source:
                    for line in source:
                        if not line.strip():
                            continue

                        record = json.loads(line)
                        contract = record.get("payload", record)
                        total += 1

                        if str(contract.get("regionCode")) != "77":
                            continue
                        if str(contract.get("fz")) != "44":
                            continue

                        row = contract_to_row(contract, filename)
                        if row is not None:
                            rows.append(row)
                            selected += 1

        file_counts.append({"month": month, "source_rows": total, "moscow_rows": selected})
        print("Записей всего:", total, "контрактов Москвы:", selected, flush=True)

    contracts = pd.DataFrame(rows)
    contracts = contracts.sort_values("published_at").drop_duplicates("contract_id", keep="last")
    contracts = contracts.sort_values(["sign_date", "contract_id"]).reset_index(drop=True)

    return contracts, pd.DataFrame(file_counts)
