.PHONY: install test lint smoke check benchmark

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

lint:
	python -m ruff check src common tests run_all.py examples tools

smoke:
	python run_all.py

check: lint test smoke

benchmark:
	python tools/benchmark_attention.py
