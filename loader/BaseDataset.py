import abc

import torch.utils.data

from loader import CsiDataloader
from loader import DataType


class BaseDataset(torch.utils.data.Dataset, abc.ABC):

    def __init__(self, csiDataloader: CsiDataloader, dataType: DataType, snr_range: list) -> None:
        super(BaseDataset, self).__init__()
        self.csiDataloader = csiDataloader
        self.dataType = dataType
        self.snr_range = snr_range
        self.h = csiDataloader.get_h(dataType)

    @abc.abstractmethod
    def cuda(self):
        pass

    @abc.abstractmethod
    def reload(self):
        pass
