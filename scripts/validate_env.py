#!/usr/bin/env python3
"""Valida o ambiente: versão do Python, deps críticas, .env e acesso ao dataset.

Falhas *hard* (versão do Python, imports faltando) retornam código != 0. As demais
(`.env`, token DagsHub, dataset) são avisos: o script passa logo após `make install`,
antes de o usuário criar o `.env` e baixar os dados.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

CHECK, CROSS, WARN = "✓", "✗", "!"
CRITICAL_IMPORTS = ("torch", "sklearn", "mlflow", "dvc", "pydantic_settings")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RATINGS_CSV = PROJECT_ROOT / "data" / "raw" / "rating.csv"


def check_python() -> tuple[bool, str, str]:
    """Confere se a versão do Python está em >=3.13,<3.14 (ver pyproject.toml)."""
    major, minor = sys.version_info[:2]
    ok = (major, minor) == (3, 13)
    hint = "" if ok else "Use Python 3.13.x (requires-python = >=3.13,<3.14)."
    return ok, f"Python {major}.{minor} (esperado 3.13.x)", hint


def check_import(module: str) -> tuple[bool, str, str]:
    """Verifica se um módulo crítico é importável no ambiente atual."""
    ok = importlib.util.find_spec(module) is not None
    hint = "" if ok else f"Dep '{module}' ausente. Rode `make install`."
    return ok, f"import {module}", hint


def check_env_file() -> tuple[bool, str, str]:
    """Avisa se o .env ainda não existe."""
    ok = (PROJECT_ROOT / ".env").exists()
    return ok, ".env presente", "" if ok else "Rode `cp .env.example .env` e preencha."


def check_token() -> tuple[bool, str, str]:
    """Avisa se DAGSHUB_TOKEN não está definido (necessário para MLflow)."""
    from recsys.config import load_settings

    ok = bool(load_settings().dagshub_token)
    return ok, "DAGSHUB_TOKEN definido", "" if ok else "Defina DAGSHUB_TOKEN no .env."


def check_dataset() -> tuple[bool, str, str]:
    """Avisa se o dataset cru (rating.csv) não está acessível."""
    ok = RATINGS_CSV.exists()
    hint = "" if ok else "Rode `make data-download` ou `make data-pull`."
    return ok, f"dataset {RATINGS_CSV.relative_to(PROJECT_ROOT)}", hint


def report(label: str, ok: bool, hint: str, hard: bool) -> None:
    """Imprime uma linha do relatório com o símbolo adequado."""
    mark = CHECK if ok else (CROSS if hard else WARN)
    tail = f"  → {hint}" if hint and not ok else ""
    print(f"  {mark} {label}{tail}")


def main() -> int:
    """Roda todos os checks; retorna 1 se algum check *hard* falhar."""
    hard = [check_python(), *(check_import(m) for m in CRITICAL_IMPORTS)]
    soft = [check_env_file(), check_token(), check_dataset()]

    print("Checks obrigatórios:")
    hard_failed = False
    for ok, label, hint in hard:
        report(label, ok, hint, hard=True)
        hard_failed = hard_failed or not ok

    print("\nAvisos (não bloqueiam):")
    for ok, label, hint in soft:
        report(label, ok, hint, hard=False)

    if hard_failed:
        print("\nAmbiente incompleto: corrija os itens marcados com ✗.", file=sys.stderr)
        return 1
    print("\nAmbiente OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
