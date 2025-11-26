
import torch
from torch.utils.data import Dataset

from sklearn.datasets import load_iris


class IrisDataset(Dataset):
    def __init__(self):
        self._iris = load_iris()

    def __len__(self):
        return len(self._iris['target'])

    def __getitem__(self, idx: int):
        return torch.Tensor(self._iris['data'][idx, :]), self._iris['target'][idx]


class BinaryIrisDataset(IrisDataset):
    def __init__(self, ref_target):
        super().__init__()
        self._ref_target = ref_target

    def __getitem__(self, idx: int):
        x, iris_target = super().__getitem__(idx)
        if iris_target == self._ref_target:
            return x, 1
        else:
            return x, -1
