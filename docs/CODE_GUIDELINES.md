# Diretrizes de código

Padrões de clean code aplicados no projeto desde a primeira linha (Etapa 1). O objetivo
é código legível, testável e com engenharia consistente.

## Idioma

- Comentários, docstrings e documentação em pt-BR, sem em-dash (usar vírgula, parênteses
  ou reescrever) e com acentuação correta.
- Termos técnicos consagrados ficam em inglês (embedding, ranking, early stopping, dot
  product, dataset, commit, etc.).
- Mensagens de log, `print`, `echo` e labels de CI podem permanecer em inglês; a tradução
  não deve alterar comportamento.

## Clean code

- **Funções curtas**: até 20 linhas, com responsabilidade única. Funções auxiliares
  privadas (prefixo `_`) para decompor lógica maior (ver `models/bpr.py`).
- **Type hints** em todas as funções públicas. O projeto usa `from __future__ import
  annotations` para anotações adiadas.
- **Docstrings** no estilo Google, descrevendo o que a função faz e seus contratos
  relevantes (por exemplo, `KeyError` para usuário desconhecido em
  `api/serving.py::recommend_for_user`).
- **Naming conventions**: `snake_case` para funções e variáveis, `PascalCase` para
  classes, nomes descritivos e sem abreviações obscuras.

## SOLID e design patterns

- **Single Responsibility**: cada módulo tem um papel claro (config, io, tracking,
  models, preprocessing, evaluation, pipeline, api). Ver [ARCHITECTURE.md](ARCHITECTURE.md).
- **Open/Closed e Liskov**: novos recomendadores estendem a ABC `Recommender`
  (`models/base.py`) sem alterar o código que os consome.
- **Dependency Inversion**: o pipeline depende da interface `Recommender` e da Factory,
  não das classes concretas.
- **Padrões aplicados**: Factory (`models/factory.py`), Strategy (`preprocessing/split.py`
  e `preprocessing/sampling.py`) e Template Method (`models/base.py::evaluate`).
  Descritos em detalhe em [ARCHITECTURE.md](ARCHITECTURE.md).

## Configuração e reprodutibilidade

- **Config externalizada**: hiperparâmetros em `configs/config.yaml` (fonte única), lidos
  via Pydantic Settings (`config.py`). Segredos e overrides de ambiente ficam no `.env` e
  têm precedência sobre o YAML.
- **Seeds fixos**: seed 42 em todo o pipeline (torch, NumPy, TruncatedSVD, amostragem).
- **Lock file**: dependências fixadas em `uv.lock`, versionado.

## Lint e formatação

O projeto usa **ruff** (config em `pyproject.toml`):

- `line-length = 100`, `target-version = "py313"`.
- Regras selecionadas: `E`, `F`, `I`, `UP`, `B`, `SIM`.

Rode o lint com:

```bash
make lint      # ruff check src tests
```

`make validate` roda lint, testes e os gates de convenção de Git; deve passar sem erros
ou avisos antes de qualquer commit.

## Estrutura de diretórios

```
src/recsys/    código do pacote (models, preprocessing, evaluation, pipeline, api)
tests/         testes pytest (smoke tests de lógica, pipeline e API)
configs/       config.yaml (hiperparâmetros)
scripts/       utilitários (validação de ambiente e de convenções, setup DVC)
data/          raw (DVC) e processed (saídas do pipeline)
models/        artefatos treinados (.pkl, .metrics.json, serving.pkl)
notebooks/     EDA e modelagem
docs/          documentação detalhada
```
