.PHONY: install train test lint up down loadtest bench

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

install:
	$(PIP) install -e ".[dev,train]"

train:
	$(PYTHON) scripts/train.py

bench:
	$(PYTHON) scripts/benchmark_runtimes.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

up:
	docker compose up --build -d

down:
	docker compose down

loadtest:
	$(PYTHON) -m locust -f loadtest/locustfile.py --headless -u 20 -r 10 -t 45s --host http://127.0.0.1:8000 --csv benchmarks/locust
