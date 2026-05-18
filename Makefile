.DEFAULT_GOAL := help

.PHONY: help install install-hooks uninstall-hooks validate validate-branch validate-commits validate-tags

UV := uv --cache-dir /tmp/uv-cache
BRANCH ?= $(shell git branch --show-current)
COMMITS_RANGE ?= origin/main..HEAD
TAGS ?=

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

validate: validate-branch validate-commits validate-tags ## Run every validation.

validate-branch: ## Validate the current branch or BRANCH=<name>.
	$(UV) run python scripts/validate_branch.py "$(BRANCH)"

validate-commits: ## Validate commits in COMMITS_RANGE, default origin/main..HEAD.
	$(UV) run python scripts/validate_commits.py --range "$(COMMITS_RANGE)"

validate-tags: ## Validate all repository tags or TAGS="1.2.3 2.0.0".
	$(UV) run python scripts/validate_tags.py $(TAGS)
