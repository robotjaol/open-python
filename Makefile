.PHONY: install test lint format notebooks catalog app clean

install:
	pip install -e ".[all,development]"

test:
	python -m pytest tests -q --ignore=tests/notebook_validation

test-notebooks:
	python -m pytest tests/notebook_validation -q

lint:
	ruff check src tests scripts tools 06_model_deployment

format:
	ruff format src tests scripts tools 06_model_deployment

notebooks:
	python scripts/validate_notebooks.py

catalog:
	python scripts/generate_catalog.py

app:
	python tools/build_app.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
