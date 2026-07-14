.DEFAULT_GOAL := help

.PHONY: help install install-hooks uninstall-hooks validate validate-env validate-branch validate-commits validate-tags \
        lint test dvc-setup data-download data-push data-pull \
        preprocess feature-eng train evaluate pipeline repro

UV  := uv --cache-dir /tmp/uv-cache
DVC := $(UV) run dvc
BRANCH ?= $(shell git branch --show-current)
COMMITS_RANGE ?= origin/main..HEAD
TAGS 	      ?=
DAGSHUB_USER  ?=
DAGSHUB_TOKEN ?=

# ── Geral ─────────────────────────────────────────────────────────────────────

help: ## Show this help message.
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make <target>\n\nTargets:\n"} /^[a-zA-Z_-]+:.*##/ {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install project dependencies, including dev tools.
	$(UV) sync --all-groups

install-hooks: ## Enable local Git hooks from .githooks.
	chmod +x .githooks/*
	git config core.hooksPath .githooks
	@echo "Git hooks installed from .githooks"

uninstall-hooks: ## Disable repository local Git hooks.
	git config --unset core.hooksPath || true
	@echo "Git hooks disabled"

# ── Validação ─────────────────────────────────────────────────────────────────

validate: lint test validate-branch validate-commits validate-tags ## Run every validation.

validate-env: ## Check Python, critical deps, .env and dataset access.
	PYTHONPATH=src $(UV) run python scripts/validate_env.py

lint: ## Lint src/ and tests/ with ruff.
	$(UV) run ruff check src tests

test: ## Run the test suite.
	$(UV) run pytest -q

validate-branch: ## Validate the current branch or BRANCH=<name>.
	$(UV) run python scripts/validate_branch.py "$(BRANCH)"

validate-commits: ## Validate commits in COMMITS_RANGE, default origin/main..HEAD.
	$(UV) run python scripts/validate_commits.py --range "$(COMMITS_RANGE)"

validate-tags: ## Validate all repository tags or TAGS="1.2.3 2.0.0".
	$(UV) run python scripts/validate_tags.py $(TAGS)

# ── DVC / Dados ───────────────────────────────────────────────────────────────

dvc-setup: ## Configure DVC remote (requires DAGSHUB_USER and DAGSHUB_TOKEN).
	@test -n "${DAGSHUB_USER}" || (echo "ERROR: DAGSHUB_USER is required.  Use: make dvc-setup DAGSHUB_USER=<user> DAGSHUB_TOKEN=<token>" && exit 1)
	@test -n "${DAGSHUB_TOKEN}" || (echo "ERROR: DAGSHUB_TOKEN is required.  Use: make dvc-setup DAGSHUB_USER=<user> DAGSHUB_TOKEN=<token>" && exit 1)
	$(DVC) remote add -f origin https://dagshub.com/JosueJNLui/fiap-mlet-challenge-fase-2.dvc
	$(DVC) config core.autostage true
	$(DVC) remote modify --local origin auth basic
	$(DVC) remote modify --local origin user ${DAGSHUB_USER}
	$(DVC) remote modify --local origin password ${DAGSHUB_TOKEN}
	@echo "DVC remote configured successfully!"

data-download: ## Download raw dataset from Kaggle and copy to data/raw/.
	$(UV) run python scripts/dvc_setup.py

data-push: ## Track data/raw with DVC and puth to DagsHub remote.
	$(DVC) add data/raw
	$(DVC) push -r origin

data-pull: ## Pull data/raw from DagsHub remote (for reproduction).
	$(DVC) pull -r origin

# ── Pipeline de ML ────────────────────────────────────────────────────────────

PY := PYTHONPATH=src $(UV) run python -m

preprocess: ## Filter/sample/reindex raw data into data/processed/.
	$(PY) recsys.pipeline.preprocess

feature-eng: ## Temporal train/test split into data/processed/.
	$(PY) recsys.pipeline.feature_eng

train: ## Train the 5 models, log MLflow runs, save to models/ (needs DagsHub creds).
	$(PY) recsys.pipeline.train

evaluate: ## Aggregate metrics into comparison.csv + metrics.json.
	$(PY) recsys.pipeline.evaluate

pipeline: preprocess feature-eng train evaluate ## Run the full pipeline end-to-end.

repro: ## Run the DVC pipeline (dvc repro).
	$(DVC) repro