from abc import abstractmethod, ABC


class Downloader(ABC):
    @abstractmethod
    def download(self, task_id: str, delete_after_download: bool = False) -> None:
        pass