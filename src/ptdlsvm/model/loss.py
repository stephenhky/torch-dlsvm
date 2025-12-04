
import torch
import torch.nn as nn


class HingeLoss(nn.Module):
    def _element_wise_loss(self, outputs: torch.Tensor, targets: torch.IntTensor):
        assert outputs.shape == targets.shape
        return torch.maximum(torch.zeros(targets.shape), 1 - outputs * targets)

    def forward(self, outputs: torch.Tensor, targets: torch.IntTensor):
        return torch.sum(self._element_wise_loss(outputs, targets))


class SVMHingeLoss(nn.Module):
    def __init__(self, c: float):
        super().__init__()
        self._hingeloss = HingeLoss()
        self._c = c

    def forward(self, w: torch.Tensor, outputs: torch.Tensor, targets: torch.IntTensor):
        hinge_loss = self._hingeloss(outputs, targets)
        return 0.5 * torch.linalg.norm(w).square() + self._c * hinge_loss
