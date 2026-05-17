# fiap-mlet-challenge-fase-2

## Gates de commit

Este projeto possui validações automatizadas para manter a rastreabilidade do
histórico Git. As validações rodam localmente via `make` e também no GitHub
Actions em pushes e pull requests.

Os gates validam:

- **Commits**: devem seguir a especificação de
  [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/#specification).
- **Branches**: devem seguir a especificação de
  [Conventional Branch 1.0.0](https://conventional-branch.github.io/#specification).
- **Tags**: devem usar versionamento semântico no formato `MAJOR.MINOR.PATCH`,
  por exemplo `1.0.0`, `1.2.3` ou `2.0.0`.

## Instalação

Este projeto usa `uv` para gerenciar dependências.

```bash
make install
```

O comando instala as dependências do projeto e as ferramentas de desenvolvimento,
incluindo `commitizen`, usado para validar as mensagens de commit.

## Como executar as validações

Para executar todos os gates:

```bash
make validate
```

Para listar todos os comandos disponíveis:

```bash
make help
```

Validações individuais:

```bash
make validate-branch
make validate-commits
make validate-tags
```

Também é possível sobrescrever os valores usados pelas validações:

```bash
make validate-branch BRANCH=feat/adicionar-pipeline
make validate-commits COMMITS_RANGE=origin/main..HEAD
make validate-tags TAGS="1.0.0 1.1.0"
```

## Padrões aceitos

Exemplos de commits válidos:

```text
feat: add prediction endpoint
fix(api): handle empty payload
docs: update setup instructions
```

Exemplos de branches válidas:

```text
main
develop
feat/adicionar-pipeline
feature/issue-123-new-login
fix/corrigir-validacao
release/v1.2.0
```

Exemplos de tags válidas:

```text
0.1.0
1.0.0
2.3.4
```

## GitHub Actions

A pipeline está em `.github/workflows/validate-conventions.yml`.

Ela executa automaticamente:

- validação de branch em pushes e pull requests;
- validação das mensagens dos commits incluídos no push ou pull request;
- validação do nome da tag quando o evento for um push de tag.
