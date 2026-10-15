import numpy as np
import pytest

from src.svm import PegasosSVM


def test_to_pm1_maps_zero_one_labels_and_stores_classes():
    model = PegasosSVM()

    encoded = model._to_pm1(np.array([0, 1, 0, 1]))

    np.testing.assert_array_equal(encoded, [-1, 1, -1, 1])
    np.testing.assert_array_equal(model.classes_, [0, 1])


@pytest.mark.parametrize(
    "labels",
    [
        np.array([]),
        np.array([0, 1, 2]),
        np.array([[0, 1], [0, 1]]),
        np.array([0, np.nan]),
    ],
)
def test_to_pm1_rejects_invalid_labels(labels):
    with pytest.raises(ValueError):
        PegasosSVM()._to_pm1(labels)