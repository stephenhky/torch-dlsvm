
from typing import Optional, Self, Union
import warnings
from os import PathLike
from collections import defaultdict

import torch
import orjson

from .utils import FeatureExtractor



class BOWFeatureExtractor(FeatureExtractor):
    def __init__(self, initial_features: Optional[dict[str, int]] = None):
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

        feature_pos_count = defaultdict(lambda : 0)
        for feature in X:
            if feature in self._features:
                feature_pos_count[self._features[feature]] += 1

        return torch.sparse_coo_tensor(
            torch.stack((
                torch.zeros(len(feature_pos_count), dtype=torch.int64),
                torch.IntTensor(list(feature_pos_count.keys()))
            ), dim=0),
            torch.FloatTensor(list(feature_pos_count.values())),
            (1, len(self._features))
        )

    def transform_from_counts(self, feature_counts: dict[str, Union[int, float]]) -> torch.Tensor:
        feature_pos_count = {
            self._features[feature]: count
            for feature, count in feature_counts.items()
        }
        return torch.sparse_coo_tensor(
            torch.stack((
                torch.zeros(len(feature_pos_count), dtype=torch.int64),
                torch.IntTensor(list(feature_pos_count.keys()))
            ), dim=0),
            torch.FloatTensor(list(feature_pos_count.values())),
            (1, len(self._features))
        )

    def __len__(self) -> int:
        return len(self._features)

    def get_feature_idx(self, feature: str) -> int:
        return self._features[feature]

    def save(self, outputpath: Union[str, PathLike]) -> None:
        open(outputpath, "wb").write(orjson.dumps(self._features))

    @classmethod
    def from_pretrained(cls, path: str | PathLike) -> Self:
        features = orjson.loads(
            open(path, "rb").read()
        )
        return BOWFeatureExtractor(features)
