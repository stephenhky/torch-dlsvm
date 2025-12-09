
import torch

from ptdlsvm.feature.extraction import BOWFeatureExtractor


def test_bow_feature_extractor() -> None:
    features = {"a": 0, "b": 1, "c": 2, "d": 3}
    bow_feature_extractor = BOWFeatureExtractor(initial_features=features)
    vector0 = bow_feature_extractor.transform(['c', 'a'])

    assert isinstance(vector0, torch.Tensor)
    assert vector0.dim() == 2
    assert vector0.is_sparse

    assert vector0[0, 0] == 1.0
    assert vector0[0, 1] == 0.0
    assert vector0[0, 2] == 1.0
    assert vector0[0, 3] == 0.0


def test_bow_feature_building() -> None:
    bow_feature_extractor = BOWFeatureExtractor()
    bow_feature_extractor.add_feature("d")
    bow_feature_extractor.add_feature("c")
    bow_feature_extractor.add_feature("a")
    bow_feature_extractor.add_feature("b")
    vector0 = bow_feature_extractor.transform(['c', 'a'])

    assert isinstance(vector0, torch.Tensor)
    assert vector0.dim() == 2
    assert vector0.is_sparse

    assert vector0[0, 0] == 0.0
    assert vector0[0, 1] == 1.0
    assert vector0[0, 2] == 1.0
    assert vector0[0, 3] == 0.0


def test_bow_feature_extractor_2():
    features = {"a": 0, "b": 1, "c": 2, "d": 3}
    bow_feature_extractor = BOWFeatureExtractor(initial_features=features)
    vector0 = bow_feature_extractor.transform(['c', 'a', 'a', 'b'])

    assert isinstance(vector0, torch.Tensor)
    assert vector0.dim() == 2
    assert vector0.is_sparse

    assert vector0[0, 0] == 2.0
    assert vector0[0, 1] == 1.0
    assert vector0[0, 2] == 1.0
    assert vector0[0, 3] == 0.0
