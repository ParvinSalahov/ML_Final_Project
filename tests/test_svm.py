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


def test_fit_learns_deterministic_separating_hyperplane():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    first = PegasosSVM(lambda_=0.01, n_iters=1000, random_state=7).fit(X, y)
    second = PegasosSVM(lambda_=0.01, n_iters=1000, random_state=7).fit(X, y)

    assert first.w is not None
    assert first.w.shape == (1,)
    assert first.b == second.b
    np.testing.assert_array_equal(first.w, second.w)
    assert np.all(y == np.where(X @ first.w + first.b >= 0, 1, 0))


@pytest.mark.parametrize("lambda_", [0, -1, np.inf, np.nan])
def test_constructor_rejects_invalid_regularization(lambda_):
    with pytest.raises(ValueError, match="lambda_"):
        PegasosSVM(lambda_=lambda_)


@pytest.mark.parametrize("n_iters", [0, -1, 1.5, True])
def test_constructor_rejects_invalid_iteration_count(n_iters):
    with pytest.raises(ValueError, match="n_iters"):
        PegasosSVM(n_iters=n_iters)


@pytest.mark.parametrize(
    ("X", "y"),
    [
        (np.array([1.0, 2.0]), np.array([0, 1])),
        (np.array([[1.0], [2.0]]), np.array([0])),
        (np.array([[1.0], [np.inf]]), np.array([0, 1])),
    ],
)
def test_fit_rejects_invalid_training_data(X, y):
    with pytest.raises(ValueError):
        PegasosSVM(n_iters=10).fit(X, y)
