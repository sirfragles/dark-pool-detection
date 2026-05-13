# ── Dark Pool Detection — Makefile ──────────────────────

.PHONY: help install test run-web run-dashboard docker-build docker-up docker-down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	pip install -r requirements.txt

test: ## Run all tests
	python3 -m pytest tests/ -v

test-fast: ## Run tests (stop on first failure)
	python3 -m pytest tests/ -v -x

test-cov: ## Run tests with coverage
	pip install -q pytest-cov
	python3 -m pytest tests/ -v --cov=src --cov-report=term-missing

run-web: ## Start Flask web interface
	python3 -m src.web.app --host 0.0.0.0 --port 5000

run-dashboard: ## Start Streamlit dashboard
	streamlit run src/dashboard/app.py --server.port 8501

run-pipeline: ## Run full detection pipeline
	python3 pipelines/run_all.py --mode sim --ticks 5000 --n-tickers 5

run-live: ## Run pipeline on live Yahoo Finance data
	python3 pipelines/run_all.py --mode live --tickers SPY QQQ AAPL

run-backtest: ## Run backtest only
	python3 pipelines/run_all.py --mode backtest --ticks 10000

docker-build: ## Build Docker image
	docker build -t dark-pool-detection:latest .

docker-up: ## Start Docker container
	docker compose up -d
	@echo "🌑 Dark Pool Detection → http://localhost:5000"

docker-down: ## Stop Docker container
	docker compose down

docker-logs: ## View Docker logs
	docker compose logs -f

docker-shell: ## Open shell in container
	docker compose exec darkpool /bin/bash

clean: ## Remove generated files
	rm -rf data/raw/yfinance/*.parquet output/*.parquet output/*.csv output/*.json output/*.jsonl output/*.txt data/models/*.joblib data/models/*.pt
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

clean-all: clean ## Remove everything generated including caches
	rm -rf data/raw output data/models .pytest_cache
