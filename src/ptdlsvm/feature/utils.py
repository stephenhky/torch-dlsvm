
from abc import abstractmethod
from typing import Any, Union

import torch


class FeatureExtractor:
    @abstractmethod
    def transform(self, X: Any) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def transform_from_counts(self, feature_counts: dict[str, Union[int, float]]) -> torch.Tensor:
        raise NotImplementedError()
