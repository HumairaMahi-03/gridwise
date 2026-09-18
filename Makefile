# GridWise — common tasks. Run `make help` for the list.
.DEFAULT_GOAL := help
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help setup test validate validate-live serve docker docker-run clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtualenv, install deps, seed .env
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt
	@[ -f .env ] || cp .env.example .env
	@echo "Ready. Add your LLM_API_KEY to .env, then: make serve"

test: ## Run the full test suite
	$(PY) -m pytest -q

validate: ## Replay the 10 public cases offline (no API key needed)
	$(PY) scripts/validate_cases.py

validate-live: ## Replay the 10 public cases using the real LLM
	$(PY) scripts/validate_cases.py --local

serve: ## Start the API on :8000 with reload
	$(VENV)/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker: ## Build the Docker image
	docker build -t gridwise .

docker-run: ## Run the container on :8000
	docker run --rm -p 8000:8000 --env-file .env gridwise

clean: ## Remove caches and the virtualenv
	rm -rf $(VENV) .pytest_cache **/__pycache__ */**/__pycache__
