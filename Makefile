.DEFAULT_GOAL := help

.PHONY: help install install-hooks uninstall-hooks validate validate-env validate-branch validate-commits validate-tags \
        lint test dvc-setup data-download data-push data-pull \
        preprocess feature-eng train evaluate pipeline repro api \
        docker-build docker-train docker-mlflow

UV  := uv --cache-dir /tmp/uv-cache
DVC := $(UV) run dvc
BRANCH ?= $(shell git branch --show-current)
COMMITS_RANGE ?= origin/main..HEAD
TAGS 	      ?=
DAGSHUB_USER  ?=
DAGSHUB_TOKEN ?=

# ── Geral ─────────────────────────────────────────────────────────────────────

help: ## Mostra esta mensagem de ajuda.
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make <target>\n\nTargets:\n"} /^[a-zA-Z_-]+:.*##/ {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Instala as dependências do projeto, incluindo ferramentas de dev.
	$(UV) sync --all-groups

install-hooks: ## Habilita os hooks locais de Git em .githooks.
	chmod +x .githooks/*
	git config core.hooksPath .githooks
	@echo "Git hooks installed from .githooks"

uninstall-hooks: ## Desabilita os hooks locais de Git do repositório.
	git config --unset core.hooksPath || true
	@echo "Git hooks disabled"

# ── Validação ─────────────────────────────────────────────────────────────────

validate: lint test validate-branch validate-commits validate-tags ## Roda todas as validações.

validate-env: ## Verifica Python, deps críticas, .env e acesso ao dataset.
	PYTHONPATH=src $(UV) run python scripts/validate_env.py

lint: ## Roda o lint de src/ e tests/ com ruff.
	$(UV) run ruff check src tests

test: ## Roda a suíte de testes.
	$(UV) run pytest -q

validate-branch: ## Valida a branch atual ou BRANCH=<nome>.
	$(UV) run python scripts/validate_branch.py "$(BRANCH)"

validate-commits: ## Valida os commits em COMMITS_RANGE, padrão origin/main..HEAD.
	$(UV) run python scripts/validate_commits.py --range "$(COMMITS_RANGE)"

validate-tags: ## Valida todas as tags do repositório ou TAGS="1.2.3 2.0.0".
	$(UV) run python scripts/validate_tags.py $(TAGS)

# ── DVC / Dados ───────────────────────────────────────────────────────────────

dvc-setup: ## Configura o remote do DVC (requer DAGSHUB_USER e DAGSHUB_TOKEN).
	@test -n "${DAGSHUB_USER}" || (echo "ERROR: DAGSHUB_USER is required.  Use: make dvc-setup DAGSHUB_USER=<user> DAGSHUB_TOKEN=<token>" && exit 1)
	@test -n "${DAGSHUB_TOKEN}" || (echo "ERROR: DAGSHUB_TOKEN is required.  Use: make dvc-setup DAGSHUB_USER=<user> DAGSHUB_TOKEN=<token>" && exit 1)
	$(DVC) remote add -f origin https://dagshub.com/JosueJNLui/fiap-mlet-challenge-fase-2.dvc
	$(DVC) config core.autostage true
	$(DVC) remote modify --local origin auth basic
	$(DVC) remote modify --local origin user ${DAGSHUB_USER}
	$(DVC) remote modify --local origin password ${DAGSHUB_TOKEN}
	@echo "DVC remote configured successfully!"

data-download: ## Baixa o dataset raw do Kaggle e copia para data/raw/.
	$(UV) run python scripts/dvc_setup.py

data-push: ## Versiona data/raw com DVC e envia para o remote DagsHub.
	$(DVC) add data/raw
	$(DVC) push -r origin

data-pull: ## Baixa data/raw do remote DagsHub (para reprodução).
	$(DVC) pull -r origin

# ── Pipeline de ML ────────────────────────────────────────────────────────────

PY := PYTHONPATH=src $(UV) run python -m

preprocess: ## Filtra/amostra/reindexa os dados raw em data/processed/.
	$(PY) recsys.pipeline.preprocess

feature-eng: ## Split temporal treino/teste em data/processed/.
	$(PY) recsys.pipeline.feature_eng

train: ## Treina os 5 modelos, loga runs no MLflow, salva em models/ (requer creds DagsHub).
	$(PY) recsys.pipeline.train

evaluate: ## Agrega as métricas em comparison.csv + metrics.json.
	$(PY) recsys.pipeline.evaluate

pipeline: preprocess feature-eng train evaluate ## Roda o pipeline completo de ponta a ponta.

repro: ## Roda o pipeline do DVC (dvc repro).
	$(DVC) repro

api: ## Sobe o modelo via FastAPI em http://localhost:8000 (requer models/).
	PYTHONPATH=src $(UV) run uvicorn recsys.api.app:app --host 0.0.0.0 --port 8000

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build: ## Constrói a imagem da aplicação (recsys:local).
	docker build -t recsys:local .

docker-train: ## Roda o pipeline completo em um container (requer .env + data/raw).
	docker compose run --rm train

docker-mlflow: ## Sobe a UI local do MLflow em http://localhost:5000.
	docker compose up mlflow