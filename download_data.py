"""Загружает архивы Clearspending и не скачивает уже существующие файлы."""

from urllib.request import urlretrieve

from config import RAW, MONTHS


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    base_url = "https://old.clearspending.ru/download/opendata/"

    for month in MONTHS:
        filename = "contracts_44fz_" + month + "-latest.jsonl.zip"
        destination = RAW / filename
        if destination.exists():
            print("Уже есть:", filename)
            continue

        print("Загрузка:", filename, flush=True)
        temporary = RAW / (filename + ".part")
        urlretrieve(base_url + filename, temporary)
        temporary.rename(destination)
        print("Готово:", filename, flush=True)
