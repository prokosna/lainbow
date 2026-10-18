from logging.config import fileConfig

from alembic import context
from sqlmodel import SQLModel

import domain.schemas  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata

APP_TABLES = {"song", "songfeatures", "songembedding", "taskresult"}


def include_object(
    obj: object,
    name: str | None,
    type_: str,
    reflected: bool,
    compare_to: object,
) -> bool:
    return not (type_ == "table" and name not in APP_TABLES)


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is None:
        raise RuntimeError("No connection provided in config.attributes['connection']")

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


run_migrations_online()
