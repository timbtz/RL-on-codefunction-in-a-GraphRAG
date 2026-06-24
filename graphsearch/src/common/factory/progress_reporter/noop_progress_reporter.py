import logging

from common.factory.progress_reporter.progress_reporter_base import ProgressReporterBase

logger = logging.getLogger(__name__)


class NoopProgressReporter(ProgressReporterBase):
    def __init__(self) -> None:
        logger.warning("No progress reporter provided, using NoopProgressReporter as fallback. Progress and log messages will be silently discarded.")

    async def on_debug(self, message: str) -> None:
        pass

    async def on_info(self, message: str) -> None:
        pass

    async def on_warning(self, message: str) -> None:
        pass

    async def on_error(self, message: str) -> None:
        pass

    async def on_progress(self, current: int, total: int) -> None:
        pass
