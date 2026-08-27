"""Shared base for typed configs loaded from YAML, with environment
variable overrides layered on top (e.g. `LLM_JB_MODEL_DEVICE=cpu`) —
handy for sweep.py to vary a single field without duplicating YAML files.

Subclasses set their own `env_prefix` via `model_config` and are always
loaded with `.load(path)`, never instantiated directly from arbitrary
kwargs (config files are meant to be reviewable YAML, not code).
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

T = TypeVar("T", bound="YamlSettings")


class YamlSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid")

    @classmethod
    def load(cls: type[T], path: Path) -> T:
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")

        class _Loaded(cls):  # type: ignore[misc]
            model_config = SettingsConfigDict(**{**cls.model_config, "yaml_file": path})

            @classmethod
            def settings_customise_sources(
                cls,
                settings_cls: type[BaseSettings],
                init_settings: PydanticBaseSettingsSource,
                env_settings: PydanticBaseSettingsSource,
                dotenv_settings: PydanticBaseSettingsSource,
                file_secret_settings: PydanticBaseSettingsSource,
            ) -> tuple[PydanticBaseSettingsSource, ...]:
                return (env_settings, YamlConfigSettingsSource(settings_cls))

        return _Loaded()
