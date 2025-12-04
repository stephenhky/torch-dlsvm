
from os import PathLike
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

from .model.svm import SVM
from .model.loss import SVMHingeLoss


def train(
        dataset: Dataset,
        feature_vec_dim: int,
        nb_epochs: int,
        batchsize: int,
        c: float = 1.0,
        device: torch.device = torch.device("cpu"),
        lr: float = 1e-3,
        chkpt_folder: Optional[str | PathLike] = None,
        save_chkpt_every_turn: Optional[int] = None,
        shuffle: bool = True,
        chkpt_to_start_training: Optional[str | PathLike] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
) -> SVM:
    model = SVM(feature_vec_dim)
    if chkpt_to_start_training is not None:
        state_dict = torch.load(chkpt_to_start_training)
        model.load_state_dict(state_dict)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    criterion = SVMHingeLoss(c)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)

    for i in range(nb_epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred_y = model(x).flatten()
            loss = criterion(model.w, pred_y, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (save_chkpt_every_turn is not None) and (i % save_chkpt_every_turn == 0):
            torch.save(model.state_dict(), Path(chkpt_folder) / f"chpt_{i}.bin")

    return model
