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
    assert first.w.shape == (first.n_components,)
    assert first.b == second.b
    np.testing.assert_array_equal(first.w, second.w)
    np.testing.assert_array_equal(first.predict(X), y)
    assert first.decision_function(X).shape == (X.shape[0],)


def test_predict_restores_original_class_labels():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array(["standard", "standard", "premium", "premium"])
    model = PegasosSVM(lambda_=0.01, n_iters=1000, random_state=7).fit(X, y)

    np.testing.assert_array_equal(model.predict(X), y)


def test_predictions_require_a_fitted_model():
    with pytest.raises(RuntimeError, match="fit"):
        PegasosSVM().predict(np.array([[0.0]]))


@pytest.mark.parametrize(
    "X",
    [
        np.array([1.0, 2.0]),
        np.array([[1.0, 2.0]]),
        np.array([[np.nan]]),
    ],
)
def test_decision_function_rejects_invalid_features(X):
    model = PegasosSVM(lambda_=0.01, n_iters=20).fit(
        np.array([[-1.0], [1.0]]), np.array([0, 1])
    )

    with pytest.raises(ValueError):
        model.decision_function(X)


def test_random_fourier_features_learn_xor_and_are_deterministic():
    X = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    options = {
        "lambda_": 0.01,
        "n_iters": 10_000,
        "random_state": 12,
        "n_components": 256,
        "gamma": 1.0,
    }

    first = PegasosSVM(**options).fit(X, y)
    second = PegasosSVM(**options).fit(X, y)

    np.testing.assert_array_equal(first.predict(X), y)
    np.testing.assert_array_equal(first.rff_weights_, second.rff_weights_)
    np.testing.assert_array_equal(first.decision_function(X), second.decision_function(X))


@pytest.mark.parametrize("n_components", [0, -1, 2.5, True])
def test_constructor_rejects_invalid_random_feature_count(n_components):
    with pytest.raises(ValueError, match="n_components"):
        PegasosSVM(n_components=n_components)


@pytest.mark.parametrize("gamma", [0, -1, np.inf, np.nan])
def test_constructor_rejects_invalid_gamma(gamma):
    with pytest.raises(ValueError, match="gamma"):
        PegasosSVM(gamma=gamma)


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
