import logging
from abc import ABC, abstractmethod

from langchain_core.vectorstores import VectorStore

from common.factory.progress_reporter.noop_progress_reporter import NoopProgressReporter
from common.factory.progress_reporter.progress_reporter_base import ProgressReporterBase


class SearchService(ABC):
    @abstractmethod
    def search(self, query: str) -> str:
        pass

    @abstractmethod
    async def asearch(
        self,
        query: str,
        reporter: ProgressReporterBase = NoopProgressReporter(),
    ) -> str:
        reporter.on_info("Starting search...")
        result = self.search(query)
        reporter.on_info("Search completed.")
        return result

