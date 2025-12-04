
from typing import Union

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from sklearn.datasets import load_iris


class IrisDataset(Dataset):
    def __init__(self):
        self._iris = load_iris()

    def __len__(self) -> int:
        return len(self._iris['target'])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Union[int, npt.NDArray[np.int64]]]:
        return torch.Tensor(self._iris['data'][idx, :]), self._iris['target'][idx]


class BinaryIrisDataset(IrisDataset):
    def __init__(self, ref_target):
        super().__init__()
        self._ref_target = ref_target

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Union[int, npt.NDArray[np.int64]]]:
        x, iris_target = super().__getitem__(idx)
        return x, np.where(iris_target == self._ref_target, 1, -1)

        # if iris_target == self._ref_target:
        #     return x, 1
        # else:
        #     return x, -1

        # return x, torch.where(torch.IntTensor(iris_target).eq(self._ref_target), 1, -1)
