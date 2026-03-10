from __future__ import annotations

import logging
import logging.config

from common.settings import read_logging_config


def configure_logging() -> None:
    """Configura logging a partir de `config/logging.yaml`, com fallback simples."""
    config = read_logging_config()
    if config:
        logging.config.dictConfig(config)
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )