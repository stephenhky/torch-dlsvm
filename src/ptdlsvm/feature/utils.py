from abc import abstractmethod

import torch


class FeatureExtractor:
    @abstractmethod
    def transform(self, X) -> torch.Tensor:
        raise NotImplementedError()
