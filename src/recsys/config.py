"""Configuração central via Pydantic Settings (configs/config.yaml sobreposto por .env).

Fonte única de verdade para caminhos e hiperparâmetros. O YAML guarda os valores
reprodutíveis (também lidos pelo DVC como ``params``); o ``.env`` sobrepõe valores de
ambiente/secret (credenciais DagsHub, URIs). Precedência: init > env > .env > YAML.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class Paths(BaseModel):
    """Diretórios de dados e artefatos."""

    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    models: Path = Path("models")


class DataCfg(BaseModel):
    """Filtros de atividade e amostragem do dataset."""

    min_user_ratings: int = 20
    min_item_ratings: int = 20
    n_users_sample: int = 20000
    test_frac: float = 0.2


class EvalCfg(BaseModel):
    """Parâmetros do protocolo de avaliação (ranking em catálogo completo)."""

    like_threshold: float = 4.0
    ranking_n_users: int = 500
    ranking_item_batch: int = 2048


class MlflowCfg(BaseModel):
    """Configuração de tracking do MLflow."""

    experiment_name: str = "MovieLens-Reco-Etapa2-Modelagem"


class Settings(BaseSettings):
    """Configuração do pipeline: YAML como base, ``.env``/ambiente sobrepõem."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        yaml_file="configs/config.yaml",
        extra="ignore",
    )

    seed: int = 42
    paths: Paths = Field(default_factory=Paths)
    data: DataCfg = Field(default_factory=DataCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
    mlflow: MlflowCfg = Field(default_factory=MlflowCfg)
    models: dict[str, dict] = Field(default_factory=dict)

    # Ambiente/secret — sem default no YAML, vêm do .env (ver Task 2).
    dagshub_repo_owner: str = "JosueJNLui"
    dagshub_repo_name: str = "fiap-mlet-challenge-fase-2"
    dagshub_user: str | None = None
    dagshub_token: str | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Adiciona o YAML como fonte de menor precedência."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache
def load_settings() -> Settings:
    """Carrega (e memoiza) a configuração do pipeline."""
    return Settings()
