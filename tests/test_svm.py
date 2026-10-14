import numpy as np
import pytest

from src.svm import PegasosSVM


@pytest.mark.parametrize(
    ("labels", "expected_classes", "expected_encoded"),
    [
        ([0, 1, 0, 1], [0, 1], [-1, 1, -1, 1]),
        (["standard", "premium", "standard"], ["premium", "standard"], [1, -1, 1]),
        ([-1, 1, -1], [-1, 1], [-1, 1, -1]),
    ],
)
def test_to_pm1_maps_two_classes_and_stores_mapping(
    labels, expected_classes, expected_encoded
):
    model = PegasosSVM()

    encoded = model._to_pm1(labels)

    np.testing.assert_array_equal(model.classes_, expected_classes)
    np.testing.assert_array_equal(encoded, expected_encoded)


@pytest.mark.parametrize("labels", [[], [1], [0, 1, 2], [[0, 1], [0, 1]], [0, np.nan]])
def test_to_pm1_rejects_invalid_labels(labels):
    model = PegasosSVM()

    with pytest.raises(ValueError):
        model._to_pm1(labels)
