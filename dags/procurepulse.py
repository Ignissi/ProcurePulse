"""Цепочка Airflow: подготовка данных → обучение → построение очереди."""

from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT / ".venv" / "bin" / "python"

with DAG(
    dag_id="procurepulse",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
) as dag:
    prepare = BashOperator(
        task_id="prepare",
        cwd=str(PROJECT),
        bash_command=f'"{PYTHON}" -m procurepulse.prepare',
    )
    train = BashOperator(
        task_id="train",
        cwd=str(PROJECT),
        bash_command=f'"{PYTHON}" -m procurepulse.train',
    )
    predict = BashOperator(
        task_id="predict",
        cwd=str(PROJECT),
        bash_command=f'"{PYTHON}" -m procurepulse.predict',
    )
    prepare >> train >> predict
