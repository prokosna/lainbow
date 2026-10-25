"""Typed settings and environment loading for Lainbow server and engine."""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from lainbow.core.method import AnalysisMethod
from lainbow.core.models import ModelId

ENV_PREFIX = "LAINBOW_"

logger = logging.getLogger(__name__)


class _LainbowSettings(BaseSettings):
    """Base settings class with common prefix and logging configuration."""

    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX)

    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="info",
        description="Log level.",
    )


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
    models: Annotated[list[ModelId], NoDecode] = Field(
        default=[ModelId.MUQ, ModelId.MUQ_MULAN],
        description="Models to analyze. Optional: clap, mert, acoustic.",
    )
    muq_max_sec: int = Field(
        default=180,
        ge=0,
        description="MuQ: seconds from the start of a track to analyze (0 = whole track).",
    )
    muq_drop_tail: bool = Field(
        default=False,
        description="MuQ: drop the last min(30 s, 15%) of tracks of 60 s or longer.",
    )
    muq_mulan_max_sec: int = Field(
        default=600,
        ge=0,
        description="MuQ-MuLan: seconds from the start of a track to analyze (0 = whole track).",
    )
    muq_mulan_drop_tail: bool = Field(
        default=False,
        description="MuQ-MuLan: drop the last min(30 s, 15%) of tracks of 60 s or longer.",
    )
    clap_max_sec: int = Field(
        default=600,
        ge=0,
        description="CLAP: seconds from the start of a track to analyze (0 = whole track).",
    )
    clap_drop_tail: bool = Field(
        default=False,
        description="CLAP: drop the last min(30 s, 15%) of tracks of 60 s or longer.",
    )
    mert_max_sec: int = Field(
        default=600,
        ge=0,
        description="MERT: seconds from the start of a track to analyze (0 = whole track).",
    )
    mert_drop_tail: bool = Field(
        default=False,
        description="MERT: drop the last min(30 s, 15%) of tracks of 60 s or longer.",
    )
    acoustic_max_sec: int = Field(
        default=600,
        ge=0,
        description="Acoustic: seconds from the start of a track to analyze (0 = whole track).",
    )
    acoustic_drop_tail: bool = Field(
        default=False,
        description="Acoustic: drop the last min(30 s, 15%) of tracks of 60 s or longer.",
    )

    @field_validator("models", mode="before")
    @classmethod
    def _parse_models(cls, v: object) -> object:
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    def method_for(self, model: ModelId) -> AnalysisMethod:
        """Return the analysis method configured for the given model."""
        match model:
            case ModelId.MUQ:
                max_sec = self.muq_max_sec
                drop_tail = self.muq_drop_tail
            case ModelId.MUQ_MULAN:
                max_sec = self.muq_mulan_max_sec
                drop_tail = self.muq_mulan_drop_tail
            case ModelId.CLAP:
                max_sec = self.clap_max_sec
                drop_tail = self.clap_drop_tail
            case ModelId.MERT:
                max_sec = self.mert_max_sec
                drop_tail = self.mert_drop_tail
            case ModelId.ACOUSTIC:
                max_sec = self.acoustic_max_sec
                drop_tail = self.acoustic_drop_tail
        return AnalysisMethod(
            max_sec=None if max_sec == 0 else max_sec,
            drop_tail=drop_tail,
        )


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


@dataclass(frozen=True)
class SettingRow:
    """Represents a single setting line for logging and description."""

    env_var: str
    value: str
    is_default: bool
    description: str


def describe(settings: BaseSettings) -> list[SettingRow]:
    """Describe all settings in declaration order with secrets masked."""
    rows: list[SettingRow] = []
    base_fields = set(_LainbowSettings.model_fields)
    field_names = sorted(type(settings).model_fields, key=lambda f: f in base_fields)
    for field_name in field_names:
        field = type(settings).model_fields[field_name]
        env_var = f"{ENV_PREFIX}{field_name}".upper()
        val = getattr(settings, field_name)
        if isinstance(val, SecretStr):
            val_str = "********"
        elif isinstance(val, list):
            val_str = ",".join(str(item) for item in val)
        else:
            val_str = str(val)
        is_default = field_name not in settings.model_fields_set
        desc = field.description or ""
        rows.append(
            SettingRow(
                env_var=env_var,
                value=val_str,
                is_default=is_default,
                description=desc,
            )
        )
    return rows


class SettingsError(Exception):
    """Raised when settings are invalid or missing."""


def load_settings[S: BaseSettings](cls: type[S]) -> S:
    """Load settings from environment variables, raising SettingsError on failure."""
    try:
        return cls()
    except ValidationError as e:
        lines: list[str] = []
        for err in e.errors():
            loc_name = str(err["loc"][0]) if err["loc"] else ""
            field = cls.model_fields.get(loc_name)
            env_var = f"{ENV_PREFIX}{loc_name}".upper()
            msg = "missing" if err["type"] == "missing" else err["msg"]
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
