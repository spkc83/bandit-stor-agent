.PHONY: install install-dev test lint format smoke

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,obp]"

test:
	pytest -q

lint:
	ruff check src tests scripts

format:
	black src tests scripts
	ruff check --fix src tests scripts

smoke:
	python scripts/run_full_pipeline.py data=tiny_fixture
