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


def test_fit_learns_separable_data_deterministically():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    first = PegasosSVM(
        lambda_=0.01,
        n_iters=1000,
        random_state=7,
    ).fit(X, y)

    second = PegasosSVM(
        lambda_=0.01,
        n_iters=1000,
        random_state=7,
    ).fit(X, y)

    assert first.w.shape == (1,)
    np.testing.assert_array_equal(first.w, second.w)
    assert first.b == second.b