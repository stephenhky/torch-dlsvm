
# Reference: https://bytepawn.com/svm-with-pytorch.html

import torch


class SVM(torch.nn.Module):
    def __init__(self, nbfeatures: int):
        super().__init__()
        self.nbfeatures = nbfeatures
        self.w = torch.nn.Parameter(torch.rand(self.nbfeatures), requires_grad=True)
        self.b = torch.nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x: torch.Tensor):
        return torch.dot(x, self.w.T) - self.b
