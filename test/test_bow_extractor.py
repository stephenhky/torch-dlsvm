
import torch

from ptdlsvm.feature.extraction import BOWFeatureExtractor


def test_bow_feature_extractor() -> None:
    features = {"a": 0, "b": 1, "c": 2, "d": 3}
    bow_feature_extractor = BOWFeatureExtractor(initial_features=features)
    vector0 = bow_feature_extractor.transform(['c', 'a'])
    assert isinstance(vector0, torch.Tensor)
    assert isinstance(vector0, torch.sparse_coo_tensor)
    assert vector0.dim() == 2

