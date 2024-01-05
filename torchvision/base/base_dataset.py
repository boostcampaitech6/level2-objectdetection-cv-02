from torch.utils.data import Dataset
from abc import abstractmethod


class BaseDatset(Dataset):
    @abstractmethod
    def __getitem__(self, *inputs):
        """
        Forward pass logic
        get dataset by index

        :return: data output
        """
        raise NotImplementedError


