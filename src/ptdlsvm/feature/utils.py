
from abc import abstractmethod
from typing import Any

import torch


class FeatureExtractor:
    @abstractmethod
    def transform(self, X: Any) -> torch.Tensor:
        raise NotImplementedError()
