import logging

from gpustack.config.config import Config
from gpustack.logging import setup_logging

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, cfg: Config):
        self._config = cfg

    def start(self):
        setup_logging(self._config.debug)
        logger.info("Starting GPUStack worker.")