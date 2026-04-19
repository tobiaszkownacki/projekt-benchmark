from abc import ABC, abstractmethod

from torch.utils.data.dataset import ConcatDataset


class Dataset(ABC):
    @abstractmethod
    def get(self) -> ConcatDataset:
        pass
