
from typing import Union

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
from sklearn.datasets import load_iris
import polars as pl

from ptdlsvm.train import train


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


def test_training() -> None:
    iris_ds = IrisDataset()

    bin0_iris_ds = BinaryIrisDataset(0)
    bin1_iris_ds = BinaryIrisDataset(1)
    bin2_iris_ds = BinaryIrisDataset(2)

    device = torch.device("cpu")

    model0 = train(bin0_iris_ds, 4, 100, 5, c=10.0, device=device)
    model1 = train(bin1_iris_ds, 4, 100, 5, c=10.0, device=device)
    model2 = train(bin2_iris_ds, 4, 100, 5, c=10.0, device=device)

    pred_y_0 = model0(iris_ds[:][0].to(device)).flatten()
    pred_y_1 = model1(iris_ds[:][0].to(device)).flatten()
    pred_y_2 = model2(iris_ds[:][0].to(device)).flatten()

    result_df = pl.DataFrame({
        "ref_label": iris_ds[:][1],
        "model0_label": pred_y_0.detach().cpu(),
        "model1_label": pred_y_1.detach().cpu(),
        "model2_label": pred_y_2.detach().cpu()
    })

    nbdata = len(result_df)
    for i in range(3):
        tp = np.sum(np.array(result_df['ref_label']==i) & np.array(result_df['model0_label']>0))
        fp = np.sum(np.array(result_df['ref_label']!=i) & np.array(result_df['model0_label']>0))
        tn = np.sum(np.array(result_df['ref_label']!=i) & np.array(result_df['model0_label']<0))
        fn = np.sum(np.array(result_df['ref_label']==i) & np.array(result_df['model0_label']<0))
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1score = 2*recall*precision/(recall+precision)
        accuracy = (tp + tn) / nbdata
        print(f"Model {i}")
        print(f" recall: {recall*100:.2f}%")
        print(f" precision: {precision*100:.2f}%")
        print(f" F1-score: {f1score*100:.2f}%")
        print(f" accuracy: {accuracy*100:.2f}%")
