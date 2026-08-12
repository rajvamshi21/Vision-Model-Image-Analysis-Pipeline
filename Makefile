.PHONY: help install dev db-up db-down init-db demo-data ingest bench api ui test lint clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Minimal install (numpy + pillow only)
	pip install -e .

dev:      ## Full install with db, api, demo and dev extras
	pip install -e ".[db,api,demo,dev]"

db-up:    ## Start Postgres + pgvector
	docker compose up -d db

db-down:  ## Stop and remove containers
	docker compose down

init-db:  ## Create schema and indexes
	vqa init-db

demo-data: ## Generate the labelled synthetic product-photo set
	python demo/generate_sample_images.py --out data/sample --count 120

ingest:   ## Analyse and index data/sample into pgvector
	vqa ingest data/sample --sku-from-name

bench:    ## Reproduce docs/benchmarks.md (no database required)
	python demo/benchmark.py --data data/sample

api:      ## Run the REST service
	uvicorn vqa.api:app --reload --port 8000

ui:       ## Run the Streamlit demo
	streamlit run demo/app.py

test:
	pytest -q

lint:
	ruff check src tests demo

clean:
	rm -rf .pytest_cache .ruff_cache data/sample **/__pycache__
