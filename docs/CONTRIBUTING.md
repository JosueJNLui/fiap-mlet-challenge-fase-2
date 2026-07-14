# Guia de contribuição

Como preparar o ambiente, seguir as convenções e validar o trabalho antes de commitar.

## Pré-requisitos

- Python 3.13 (o projeto exige `>=3.13,<3.14`).
- [`uv`](https://docs.astral.sh/uv/) para gerenciar dependências.
- Credenciais DagsHub (para treino, tracking MLflow e remote DVC).

## Setup do ambiente

```bash
make install            # uv sync --all-groups (deps de prod + dev)
cp .env.example .env     # preencha DAGSHUB_TOKEN / DAGSHUB_USER
make validate-env        # confere Python, deps críticas, .env e acesso ao dataset
```

## Fluxo de trabalho

```bash
make test               # roda a suíte de testes (pytest)
make pipeline           # roda o pipeline completo (preprocess -> ... -> evaluate)
make repro              # equivalente via DVC (dvc repro)
make api                # sobe a API FastAPI em http://localhost:8000
```

Consulte `make help` para todos os targets disponíveis.

## Convenções de Git

O projeto valida branches, commits e tags de forma automatizada. As regras completas
estão em [AGENTS.md](../AGENTS.md); em resumo:

- **Branches**: Conventional Branch 1.0.0, no formato `<tipo>/<descricao>` (tipos:
  `feature`, `feat`, `bugfix`, `fix`, `hotfix`, `release`, `chore`), além das trunk
  `main`/`master`/`develop`.
- **Commits**: Conventional Commits 1.0.0, no formato `<tipo>[escopo]: <descricao>`.
- **Tags**: SemVer estrito `MAJOR.MINOR.PATCH` (por exemplo `1.2.3`).

### Validação obrigatória

Antes de considerar qualquer trabalho concluído, rode:

```bash
make validate           # lint + testes + validação de branch, commits e tags
```

`make validate` deve passar sem erros ou avisos. Validações individuais e overrides:

```bash
make validate-branch BRANCH=feat/adicionar-pipeline
make validate-commits COMMITS_RANGE=origin/main..HEAD
make validate-tags TAGS="1.0.0 1.1.0"
```

### Hooks locais (opcionais)

```bash
make install-hooks      # habilita os hooks versionados em .githooks/
make uninstall-hooks    # desabilita
```

Hooks disponíveis: `pre-commit` (nome da branch), `commit-msg` (mensagem do commit),
`pre-push` (branch, commits e tags enviados).

## Padrão de código e idioma

- Siga as [diretrizes de código](CODE_GUIDELINES.md): funções curtas, type hints,
  docstrings Google style, ruff sem erros.
- Comentários, docstrings e documentação em pt-BR, sem em-dash e com acentuação; termos
  técnicos permanecem em inglês. Mensagens/logs/echos podem ficar em inglês (não devem
  alterar comportamento).
