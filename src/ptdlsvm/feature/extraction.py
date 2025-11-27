
from typing import Optional, Self
import warnings
from os import PathLike

import torch
import orjson

from .utils import FeatureExtractor



class BOWFeatureExtractor(FeatureExtractor):
    def _init__(self, initial_features: Optional[dict[str, int]] = None):
        if initial_features is None:
            self._features = {}
        else:
            self._features = initial_features

    def add_feature(self, feature: str) -> None:
        if feature in self._features:
            warnings.warn(f"Feature '{feature}' is already present.")
        else:
            self._features[feature] = len(self._features)

    def transform(self, X: list[str] | str) -> torch.Tensor:
        if isinstance(X, str):
            X = [X]

        feature_pos = [
            self._features[feature] for feature in X if feature in self._features
        ]

        return torch.sparse_coo_tensor(
            torch.stack((
                torch.zeros(len(feature_pos), dtype=torch.int64),
                torch.IntTensor(feature_pos)
            ), dim=0),
            torch.ones(len(feature_pos)),
            (1, len(self._features))
        )

    def __len__(self) -> int:
        return len(self._features)

    @classmethod
    def from_pretrained(cls, path: str | PathLike) -> Self:
        features = orjson.loads(
            open(path, "rb").read()
        )
        return BOWFeatureExtractor(features)
