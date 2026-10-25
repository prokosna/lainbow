"""Typed settings and environment loading for Lainbow server and engine."""

import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AnyHttpUrl, Field, SecretStr, ValidationError, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from lainbow.core import models
from lainbow.core.method import AnalysisMethod

ENV_PREFIX = "LAINBOW_"

logger = logging.getLogger(__name__)


def _models_description() -> str:
    """Return the formatted description of available models derived from MODELS."""
    return f"Models to analyze. Available: {', '.join(models.MODELS.keys())}."


def method_env_var(model: str, field: str) -> str:
    """Return the environment variable name for a per-model analysis method field."""
    return f"{ENV_PREFIX}{model.upper()}_{field.upper()}"


class MethodSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source that loads per-model analysis methods from environment."""

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        methods_data: dict[str, dict[str, Any]] = {}
        explicit_vars: set[str] = set()
        for model_key in models.MODELS:
            model_dict: dict[str, Any] = {}
            for field_name in AnalysisMethod.model_fields:
                env_key = method_env_var(str(model_key), field_name)
                if env_key in os.environ:
                    model_dict[field_name] = os.environ[env_key]
                    explicit_vars.add(env_key)
            methods_data[str(model_key)] = model_dict
        return {
            "methods": methods_data,
            "explicit_method_vars": frozenset(explicit_vars),
        }


@dataclass(frozen=True)
class SettingRow:
    """Represents a single setting line for logging and description."""

    env_var: str
    value: str
    is_default: bool
    description: str


class _LainbowSettings(BaseSettings):
    """Base settings class with common prefix and logging configuration."""

    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX)

    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="info",
        description="Log level.",
    )

    def _field_row(self, field_name: str, field_info: FieldInfo) -> SettingRow:
        env_var = f"{ENV_PREFIX}{field_name}".upper()
        val = getattr(self, field_name)
        if isinstance(val, SecretStr):
            val_str = "********"
        elif isinstance(val, list):
            val_str = ",".join(str(item) for item in val)
        else:
            val_str = str(val)
        is_default = field_name not in self.model_fields_set
        return SettingRow(
            env_var=env_var,
            value=val_str,
            is_default=is_default,
            description=field_info.description or "",
        )

    def _generic_field_rows(self) -> list[SettingRow]:
        """Return generic SettingRows for non-base, non-excluded fields."""
        base_fields = set(_LainbowSettings.model_fields)
        rows: list[SettingRow] = []
        for field_name, field_info in type(self).model_fields.items():
            if field_info.exclude or field_name in base_fields:
                continue
            rows.append(self._field_row(field_name, field_info))
        return rows

    def _base_field_rows(self) -> list[SettingRow]:
        """Return SettingRows for base settings fields (e.g. log_level)."""
        return [
            self._field_row(field_name, type(self).model_fields[field_name])
            for field_name in _LainbowSettings.model_fields
        ]

    def setting_rows(self) -> list[SettingRow]:
        """Return setting rows for this configuration in presentation order."""
        return self._generic_field_rows() + self._base_field_rows()


class ServerSettings(_LainbowSettings):
    """Configuration settings for the Lainbow server."""

    database_url: SecretStr = Field(description="PostgreSQL connection URL.")
    music_dir: Path = Field(
        default=Path("/music"),
        description="Root of the music library, the same folder as MPD's music_directory.",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Address the HTTP server listens on.",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port the HTTP server listens on.",
    )
    scan_interval_min: int = Field(
        default=60, ge=0, description="Minutes between automatic library scans (0 = never)."
    )
    local_engine_token: SecretStr | None = Field(
        default=None,
        description="Token of the engine that runs next to the server (written by `lainbow init`).",
    )
    lease_sec: int = Field(
        default=900,
        ge=60,
        description="Seconds an engine may hold a track before another engine may take it.",
    )
    text_engine_url: AnyHttpUrl = Field(
        default=AnyHttpUrl("http://engine:8001"),
        description=(
            "URL of the engine that turns search text into vectors (the engine next to the server)."
        ),
    )
    models: Annotated[list[str], NoDecode] = Field(
        default=["muq", "muq_mulan"],
        description=_models_description(),
    )
    methods: Mapping[str, AnalysisMethod] = Field(
        default_factory=lambda: {str(k): spec.default_method for k, spec in models.MODELS.items()},
        exclude=True,
    )
    explicit_method_vars: frozenset[str] = Field(
        default=frozenset(),
        exclude=True,
    )

    @field_validator("models", mode="before")
    @classmethod
    def _parse_models(cls, v: object) -> object:
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    @field_validator("models")
    @classmethod
    def _validate_models(cls, v: list[str]) -> list[str]:
        for name in v:
            if name not in models.MODELS:
                raise ValueError(f"unknown model: {name}")
        return v

    @field_validator("methods", mode="wrap")
    @classmethod
    def _validate_methods(
        cls,
        v: object,
        handler: Any,
        info: Any,
    ) -> Mapping[str, AnalysisMethod]:
        if isinstance(v, Mapping):
            for k in v:
                if k not in models.MODELS:
                    raise ValueError(f"unknown model: {k}")
            validated = handler(v)
            res: dict[str, AnalysisMethod] = {}
            for k, spec in models.MODELS.items():
                if k in validated and isinstance(v[k], Mapping):
                    res[k] = spec.default_method.model_copy(
                        update={f: getattr(validated[k], f) for f in v[k]}
                    )
                elif k in validated:
                    res[k] = validated[k]
                else:
                    res[k] = spec.default_method
            return res
        return handler(v)  # type: ignore[no-any-return]

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            MethodSettingsSource(settings_cls),
        )

    def method_for(self, model: str) -> AnalysisMethod:
        """Return the analysis method configured for the given model."""
        return self.methods[model]

    def setting_rows(self) -> list[SettingRow]:
        """Return setting rows for this configuration in presentation order."""
        rows: list[SettingRow] = []
        rows.extend(self._generic_field_rows())

        for model_key, spec in models.MODELS.items():
            method = self.method_for(str(model_key))
            for field_name, field_info in AnalysisMethod.model_fields.items():
                env_var = method_env_var(str(model_key), field_name)
                val = getattr(method, field_name)
                val_str = str(val)
                is_default = env_var not in self.explicit_method_vars
                short_name = spec.short_name
                field_desc = field_info.description or ""
                desc = f"{short_name}: {field_desc}" if field_desc else short_name
                rows.append(
                    SettingRow(
                        env_var=env_var,
                        value=val_str,
                        is_default=is_default,
                        description=desc,
                    )
                )

        rows.extend(self._base_field_rows())
        return rows


class EngineSettings(_LainbowSettings):
    """Configuration settings for the Lainbow engine."""

    server_url: AnyHttpUrl = Field(
        description="URL of the Lainbow server, e.g. http://192.168.1.10:8000.",
    )
    token: SecretStr = Field(
        description="Engine token printed by `lainbow add-engine` on the server.",
    )
    accel: Literal["auto", "cpu", "cuda", "rocm"] = Field(
        default="auto",
        description="Accelerator for the models.",
    )
    models_dir: Path = Field(
        default=Path("/models"),
        description="Where model weights are downloaded and cached.",
    )
    music_dir: Path | None = Field(
        default=None,
        description=(
            "Optional local mount of the music library; when a track exists here "
            "it is read directly instead of downloaded."
        ),
    )
    claim_size: int = Field(default=8, ge=1, le=64, description="Tracks to lease at a time.")
    decoders: int = Field(
        default=0,
        ge=0,
        description="Parallel download+decode workers (0 = number of CPU cores, at most 8).",
    )
    text_port: int = Field(
        default=8001,
        ge=0,
        le=65535,
        description="Port for the text-embedding endpoint used by the server (0 = off).",
    )
    idle_sec: float = Field(
        default=30.0,
        gt=0,
        description="Seconds to wait before asking again when there is no work.",
    )


def describe(settings: _LainbowSettings) -> list[SettingRow]:
    """Describe all settings in declaration order with secrets masked."""
    return settings.setting_rows()


class SettingsError(Exception):
    """Raised when settings are invalid or missing."""


def load_settings[S: BaseSettings](cls: type[S]) -> S:
    """Load settings from environment variables, raising SettingsError on failure."""
    try:
        return cls()
    except ValidationError as e:
        lines: list[str] = []
        for err in e.errors():
            loc = err["loc"]
            msg = "missing" if err["type"] == "missing" else err["msg"]
            if len(loc) >= 3 and loc[0] == "methods":
                model_name = str(loc[1])
                field_name = str(loc[2])
                env_var = method_env_var(model_name, field_name)
                spec = models.MODELS.get(model_name)
                method_field = AnalysisMethod.model_fields.get(field_name)
                short_name = (
                    spec.short_name
                    if spec and getattr(spec, "short_name", None)
                    else model_name.capitalize()
                )
                field_desc = (
                    method_field.description if method_field and method_field.description else ""
                )
                desc = f"{short_name}: {field_desc}" if field_desc else short_name
                lines.append(f"{env_var}: {msg} ({desc})")
            else:
                loc_name = str(loc[0]) if loc else ""
                field = cls.model_fields.get(loc_name)
                env_var = f"{ENV_PREFIX}{loc_name}".upper()
                desc = field.description if field and field.description else ""
                if desc:
                    lines.append(f"{env_var}: {msg} ({desc})")
                else:
                    lines.append(f"{env_var}: {msg}")
        raise SettingsError("\n".join(lines)) from None


def format_settings(rows: list[SettingRow]) -> str:
    """Format setting rows as an aligned text table."""
    if not rows:
        return ""
    col1_w = max(len(r.env_var) for r in rows)
    col2_w = max(len(r.value) for r in rows)
    col3_w = max(len("default" if r.is_default else "set") for r in rows)
    lines: list[str] = []
    for r in rows:
        status = "default" if r.is_default else "set"
        lines.append(
            f"{r.env_var:<{col1_w}}  {r.value:<{col2_w}}  {status:<{col3_w}}  {r.description}"
        )
    return "\n".join(lines)


def start[S: _LainbowSettings](cls: type[S]) -> S:
    """Load settings, configure logging, and log configuration table."""
    try:
        settings = load_settings(cls)
    except SettingsError as e:
        print(e, file=sys.stderr)
        raise SystemExit(2) from None

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logger.info("settings:\n" + format_settings(describe(settings)))
    return settings
