from abc import ABC, abstractmethod


class ProgressReporterBase(ABC):
    @abstractmethod
    async def on_debug(self, message: str) -> None: ...

    @abstractmethod
    async def on_info(self, message: str) -> None: ...

    @abstractmethod
    async def on_warning(self, message: str) -> None: ...

    @abstractmethod
    async def on_error(self, message: str) -> None: ...

    @abstractmethod
    async def on_progress(self, current: int, total: int) -> None: ...
