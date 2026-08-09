import logging
import time

from clipai.config import get_settings
from clipai.database import database_is_ready
from clipai.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def run(poll_interval_seconds: float = 5.0) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    LOGGER.info("worker_started")
    while True:
        if database_is_ready(settings.database_url):
            LOGGER.info("worker_idle")
        else:
            LOGGER.error("database_unavailable")
        time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    run()
