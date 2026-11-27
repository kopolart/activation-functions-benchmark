.PHONY: install test lint benchmark figures notebook clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

benchmark:
	python -m activation_benchmark run --epochs 150 --seeds 5 --output-dir results

figures:
	python -m activation_benchmark figures --results results/results.json --output-dir results/figures

notebook:
	jupyter nbconvert --to notebook --execute notebooks/activation_functions_benchmark.ipynb --inplace

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
