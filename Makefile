.PHONY: install install-dev test test-unit test-integration test-performance lint typecheck format clean docker-build docker-run help

help:
	@echo "AeroSpatium Development Makefile"
	@echo "================================"
	@echo "make install       - Install core dependencies"
	@echo "make install-dev   - Install development dependencies"
	@echo "make test          - Run all tests"
	@echo "make test-unit     - Run unit tests only"
	@echo "make test-integration - Run integration tests"
	@echo "make test-performance  - Run performance benchmarks"
	@echo "make lint          - Run ruff linter"
	@echo "make typecheck     - Run mypy type checker"
	@echo "make format        - Format code with black"
	@echo "make docker-build  - Build Docker image"
	@echo "make docker-run    - Run API server in Docker"
	@echo "make clean         - Remove build artifacts and cache"

install:
	pip install -e ".[dev]"

install-dev:
	pip install -e ".[dev]"
	pip install pytest-xdist pytest-cov

test:
	python -m pytest tests/ -v --tb=short

test-unit:
	python -m pytest tests/unit/ -v --tb=short

test-integration:
	python -m pytest tests/integration/ -v --tb=short -m integration

test-performance:
	python -m pytest tests/performance/ -v --tb=short -m "not slow"

lint:
	ruff check src/ tests/

typecheck:
	mypy src/aerospatium --ignore-missing-imports

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	rm -rf src/aerospatium.egg-info

docker-build:
	docker build -t aerospatium:latest .

docker-run:
	docker run -p 8000:8000 --rm -v $(PWD)/data:/app/data -v $(PWD)/models:/app/models aerospatium:latest

run-api:
	uvicorn aerospatium.api.main:app --reload --host 0.0.0.0 --port 8000
