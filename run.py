"""Полный запуск: подготовка данных → обучение → очередь проверки."""

from procurepulse.prepare import prepare
from procurepulse.train import train_models
from procurepulse.predict import predict


if __name__ == "__main__":
    prepare()
    train_models()
    predict()
