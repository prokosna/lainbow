"""Pytest fixtures for database tests."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text


@pytest.fixture
def clean_db() -> str:
    """Provide a clean PostgreSQL database by dropping and recreating public schema."""
    url = os.environ.get("LAINBOW_TEST_DATABASE_URL")
    if not url:
        pytest.skip("LAINBOW_TEST_DATABASE_URL not set")

    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    engine.dispose()
    return url
