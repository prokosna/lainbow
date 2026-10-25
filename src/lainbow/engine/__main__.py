"""Runnable entry point for the Lainbow engine."""

import logging

from lainbow.core.settings import EngineSettings, start

logger = logging.getLogger(__name__)


def main() -> None:
    """Load settings, configure logging, and start the Lainbow engine."""
    start(EngineSettings)
    logger.info("engine work loop is not implemented yet")


if __name__ == "__main__":
    main()
